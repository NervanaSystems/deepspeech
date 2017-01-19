#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2015-2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
from neon import NervanaObject
import platform
import os
import numpy as np
import numpy.ctypeslib as npct
import ctypes as ct
import ctypes.util
from neon.transforms.cost import Cost
try:
    from mgpu.nervanamgpu import NervanaMGPU
except ImportError:
    pass


class CTC(Cost):

    def __init__(self, max_label_len, nout=29, blank=0):

        self.max_s = int(max_label_len)
        self.nout = nout
        self.input_warp = None
        self.y = None
        self.input_lengths = self.be.zeros(
            (self.be.bsz), dtype=np.int32)
        self.label_lengths = self.be.zeros(
            (self.be.bsz), dtype=np.int32)
        self.flat_labels = self.be.iobuf(
            (1,self.max_s), dtype=np.uint8)
        self.host_labels = np.zeros(
            self.max_s*self.be.bsz, dtype=np.uint8)
    
    def init_buffer(self, y):
        if self.input_warp is None or self.y is None or self.y is not y:
            self.y = y
            # warp-CTC requires activations.shape = (T, bsz, nout)
            input_buf = self.be.iobuf(
                (self.max_t, self.nout), parallelism=self.parallelism)
            # using -1 here for reshape instead of bsz as it makes 
            # distribution simpler when using mgpu
            input_buf_3d = input_buf.reshape((self.max_t, self.nout, -1))
            self.input_warp = input_buf_3d.dimension_reorder([0, 2, 1])
            
            # warp-CTC requires gradients.shape = (T, bsz, nout)
            grad_buf = self.be.iobuf(
                (self.max_t, self.nout), parallelism=self.parallelism)
            grad_buf_3d = grad_buf.reshape((self.max_t, self.nout, -1))
            self.grad_warp = grad_buf_3d.dimension_reorder([0, 2, 1])

            # warp-CTC requires cost.shape = (1, bsz)
            self.ctc_warp = self.be.iobuf((1, ), parallelism=self.parallelism)
  
            # neon requires gradients.shape = (nout, T*bsz)
            self.grad = self.be.iobuf(
                (self.nout, self.max_t), parallelism=self.parallelism)
            self.grad_view = self.grad.reshape(self.nout, self.max_t, -1)

            # neon requires cost.shape = (1, T*bsz)
            self.ctc_cost = self.be.iobuf(
                (self.max_t, ), parallelism=self.parallelism)
            self.ctc_cost_view = self.ctc_cost.reshape(self.max_t, -1).T

            # auxiliary data structures for mgpu
            length_buf = self.be.iobuf(
                (1, ), dtype=np.int32, parallelism=self.parallelism)
            self.input_lengths = length_buf.dimension_reorder([1, 0])
            label_len_buf = self.be.iobuf(
                (1, ), dtype=np.int32, parallelism=self.parallelism)
            self.label_lengths = label_len_buf.dimension_reorder([1, 0])

    def __call__(self, y, t):

        activations = y.reshape(self.nout, -1, self.be.bsz)
        self.max_t = activations.shape[1]
        self.init_buffer(y)

        self.grad_warp.fill(0.)
        self.ctc_warp.fill(0.)
        self.ctc_cost.fill(0.)

        # prep transcripts for flattening
        _labels = t[0].get()

        # label_lengths: minibatch worth of transcript lengths
        label_lengths = t[1]
        host_label_lens =  label_lengths.get().ravel()
        self.label_lengths.set(host_label_lens)
        
        # flatten transcripts
        start = 0
        for indx, label_len in enumerate(host_label_lens):
            end = start + label_len
            self.host_labels[start:end] = _labels[:label_len, indx]
            start = end
        # set leftover host_label slots to zero (i.e. blanks)
        self.host_labels[start:] = 0
        
        # flat_labels: minibatch worth of flattened transcripts
        self.flat_labels.set(self.host_labels.reshape(1,-1))

        # input_lengths: minibatch worth of activation lengths
        self.input_lengths[:] = t[2].T * int(activations.shape[1]) / 100
        
        # reshape activations to match warp-CTC format                                      
        self.be.copy_transpose(activations, self.input_warp, (1, 2, 0))

        # call into warp-CTC
        self.be_ctc(
            self.nout,
            self.input_warp,
            self.flat_labels,
            self.grad_warp,
            self.label_lengths,
            self.input_lengths,
            self.ctc_warp,
            self.max_s,
            self.max_t)

        # warp-ctc only returns 1 cost per example ...
        # broadcast ctc_warp (shape=(1, bsz)) to ctc_cost (shape=(1,T*bsz))
        self.ctc_cost_view[:] = self.ctc_warp.T

        return self.ctc_cost

    def be_ctc(
                self, 
                nout, 
                inputs, 
                labels, 
                grads, 
                label_lens, 
                input_lens, 
                costs, 
                max_s, 
                max_t):

        if self.be.backend_name == "gpu":
            self.be_ctc_gpu(
                inputs, 
                labels, 
                grads, 
                label_lens, 
                input_lens, 
                costs, 
                self.be.stream,
                self.be.bsz, 
                nout, 
                max_s, 
                max_t)
        elif self.be.backend_name == "mgpu":
            self.be_ctc_mgpu(
                inputs, 
                labels, 
                grads, 
                label_lens, 
                input_lens, 
                costs, 
                nout, 
                max_s, 
                max_t)
        else:
            raise NotImplementedError()

    def be_ctc_mgpu(
                self, 
                inputs, 
                labels, 
                grads, 
                label_lens, 
                input_lens, 
                costs, 
                nout, 
                max_s, 
                max_t):

        full_bsz = self.be.bsz
        dev_bsz = full_bsz // self.be.num_dev

        # Need to split flattened labels if data parallel                                               
        if self.parallelism == 'Data':
            label_lens = label_lens.get()
            dev_lens = [np.sum(label_lens[i * dev_bsz:(i + 1) * dev_bsz]) 
                        for i in range(self.be.num_dev)]
            dev_offsets = np.cumsum(dev_lens) - dev_lens
            labels.replicate(False)

            for _ctx, _stream, _t, _offset, _len in zip(self.be.ctxs, 
                                                        self.be.streams, 
                                                        labels, 
                                                        dev_offsets, 
                                                        dev_lens):
                if _t is not labels.tensorlist[0]:
                    _ctx.push()
                    self.be.stream = _stream
                    _t[:_len] = labels.tensorlist[0][_offset:(_offset + _len)]
                    _ctx.pop()

            self.be.stream = self.be.streams[0]

            sub_bszs = self.be._divide_workload(full_bsz)
            label_lens.fragment(sub_bszs, True, 0)

            assert inputs.ptype == 'fragment'
            assert labels.ptype == 'replica'
            assert grads.ptype == 'fragment'
            assert costs.ptype == 'fragment'
            assert label_lens.ptype == 'fragment'
            assert input_lens.ptype == 'fragment'
        else:
            assert inputs.ptype == 'singlenode'
            assert labels.ptype == 'singlenode'
            assert grads.ptype == 'singlenode'
            assert costs.ptype == 'singlenode'
            assert label_lens.ptype == 'singlenode'
            assert input_lens.ptype == 'singlenode'

            sub_bszs = [full_bsz]

        commargs = dict(nout=nout, max_s=max_s, max_t=max_t)
        rptargs = ['inputs', 'labels', 'grads', 'label_lens', 
                   'input_lens', 'costs', 'stream', 'bsz']
        rptvals = zip(inputs, labels, grads, label_lens, 
                                input_lens, costs, self.be.streams, sub_bszs)
        dev_args = [dict((k, v) for k, v in zip(rptargs, _vals), **commargs) 
                    for _vals in rptvals]

        for _ctx, _dev_args in zip(self.be.ctxs, dev_args):
            _ctx.push()
            self.be_ctc_gpu(**_dev_args)
            _ctx.pop()

        self.be.bsz = full_bsz
        self.be.stream = self.be.streams[0]

        if self.parallelism == 'Data':
            label_lens.swap_shadow()
            labels.swap_shadow()

    def be_ctc_gpu(
            self,
            inputs, 
            labels,
            grads,
            label_lens,
            input_lens,
            costs,
            stream,
            bsz,
            nout,
            max_s,
            max_t):
        """                                                                                             
        Calling Warp-CTC                                                                                
        """

        libpath = os.path.join(os.path.dirname(__file__), 
                                                  "..", "src", "transforms", "libwarpctc.so")
        assert os.path.isfile(libpath), "libwarpctc.so not found.  Run make"
        self.ctclib = npct.load_library(libpath, "")

        # map first function to get workspace size                                                      
        self.ctclib.get_workspace_size_gpu.restype = int
        self.ctclib.get_workspace_size_gpu.argtypes = [ct.c_int,
                                                       ct.c_int,
                                                       ct.c_int,
                                                       ct.c_int]

        scratch_size = self.ctclib.get_workspace_size_gpu(
            max_s, max_t, nout, bsz)
        self.be.set_scratch_size(scratch_size)
        workspace = self.be.scratch_buffer(scratch_size)

        # map ctc function                                                                              
        self.ctclib.compute_ctc_gpu.restype = int
        self.ctclib.compute_ctc_gpu.argtypes = [ct.POINTER(ct.c_float),
                                                ct.POINTER(ct.c_float),
                                                npct.ndpointer(
                                                    dtype=np.int32, ndim=1),
                                                npct.ndpointer(
                                                    dtype=np.int32, ndim=1),
                                                npct.ndpointer(
                                                    dtype=np.int32, ndim=1),
                                                ct.c_int,
                                                ct.c_int,
                                                ct.POINTER(ct.c_float),
                                                ct.c_void_p,
                                                ct.POINTER(ct.c_char)]

        inputs_buf = ct.cast(int(inputs.gpudata), ct.POINTER(ct.c_float))
        grads_buf = ct.cast(int(grads.gpudata), ct.POINTER(ct.c_float))
        costs_buf = ct.cast(int(costs.gpudata), ct.POINTER(ct.c_float))
        workspace_buf = ct.cast(workspace, ct.POINTER(ct.c_char))

        if stream is None:
            stream_buf = ct.cast(stream, ct.c_void_p)
        else:
            stream_buf = ct.cast(stream.handle, ct.c_void_p)

        status = self.ctclib.compute_ctc_gpu(inputs_buf, 
                                             grads_buf,
                                             np.array(labels.get().ravel(), 
                                                            dtype=np.int32),
                                             np.array(label_lens.get().ravel(), 
                                                            dtype=np.int32),
                                             np.array(input_lens.get().ravel(),
                                                      dtype=np.int32),
                                             nout, 
                                             bsz,
                                             costs_buf,
                                             stream_buf,
                                             workspace_buf)

        assert status is 0, "Warp-CTC run failed"
        return

    def bprop(self, y, t):
        # grad_warp is in (S, B, F) format, need to transpose to (F, S, B)
        # and self.grad_view is a reshaped (F, S, B) view of self.grad (F, S*B)
        self.be.copy_transpose(self.grad_warp, self.grad_view, (2, 0, 1))

        return self.grad

