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


class CTC(Cost):

    def __init__(self, max_label_len, nout=29, blank=0):

        self.max_s = int(max_label_len)
        self.nout = nout
        self.input_warp = None
        self.y = None
        self.input_lengths = self.be.zeros((self.be.bsz), dtype=np.int32)

        self.ctclib = None

    def init_buffer(self, y):

        if self.input_warp is None or self.y is None or self.y is not y:
            self.y = y
            # warp-CTC requires activations.shape = (T, bsz, nout)
            self.input_warp = self.be.zeros(
                (self.max_t, self.be.bsz, self.nout))
            # warp-CTC requires gradients.shape = (T, bsz, nout)
            self.grad_warp = self.be.zeros(
                (self.max_t, self.be.bsz, self.nout))
            # warp-CTC requires cost.shape = (1, bsz)
            self.ctc_warp = self.be.zeros((1, self.be.bsz))

            # neon requires gradients.shape = (nout, T*bsz)
            self.grad = self.be.iobuf((self.nout, self.max_t))
            self.grad_view = self.grad.reshape(
                self.nout, self.max_t, self.be.bsz)
            # neon requires cost.shape = (nout, T*bsz)
            self.ctc_cost = self.be.zeros((1, self.max_t * self.be.bsz))
            self.ctc_cost_view = self.ctc_cost.reshape(
                self.max_t, self.be.bsz).T

    def __call__(self, y, t):

        activations = y.reshape(self.nout, -1, self.be.bsz)
        self.max_t = activations.shape[1]
        self.init_buffer(y)
        self.grad_warp.fill(0.)
        self.ctc_warp.fill(0.)
        self.ctc_cost.fill(0.)
        # flat_labels: minibatch worth of transcripts
        flat_labels = t[0]

        # label_lengths: minibatch worth of transcript lengths
        label_lengths = t[1]

        # input_lengths: minibatch worth of activation lengths
        self.input_lengths[:] = t[2].T * int(activations.shape[1]) / 100

        # reshape activations to match warp-CTC format
        self.be.copy_transpose(activations, self.input_warp, (1, 2, 0))

        # call into warp-CTC
        self.be_ctc(
            self.nout,
            self.input_warp,
            flat_labels,
            self.grad_warp,
            label_lengths,
            self.input_lengths,
            self.ctc_warp)

        # warp-ctc only returns 1 cost for each example
        # broadcast ctc_warp (shape = (1,bsz)) to ctc_cost (shape=(1, T*bsz))
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
            costs):

        libpath = os.path.join(os.path.dirname(__file__),
                               "..", "src", "transforms", "libwarpctc.so")
        assert os.path.isfile(libpath), "libwarpctc.so not found.  Run make"
        self.ctclib = npct.load_library(libpath, "")

        if self.be.backend_name == "gpu":
            self.be_ctc_gpu(
                nout,
                inputs,
                labels,
                grads,
                label_lens,
                input_lens,
                costs)
        elif self.be.backend_name == "cpu" or self.be.backend_name == "mkl":
            self.be_ctc_cpu(
                inputs,
                labels,
                grads,
                label_lens,
                input_lens,
                costs,
                nout)
        else:
            raise NotImplementedError()

    def be_ctc_gpu(
            self,
            nout,
            inputs,
            labels,
            grads,
            label_lens,
            input_lens,
            costs):
        """
        Calling Warp-CTC
        """

        # Set up cuda stream
        if self.be.stream is None:
            stream_buf = ct.cast(self.be.stream, ct.c_void_p)
        else:
            stream_buf = ct.cast(
                id(self.be.stream), ct.POINTER(ct.c_void_p)).contents

        # map first function to get workspace size
        self.ctclib.get_workspace_size_gpu.restype = int
        self.ctclib.get_workspace_size_gpu.argtypes = [npct.ndpointer(dtype=np.int32, ndim=1),
                                                       npct.ndpointer(dtype=np.int32, ndim=1),
                                                       ct.c_int,
                                                       ct.c_int,
                                                       ct.c_void_p]
        scratch_size = self.ctclib.get_workspace_size_gpu(np.array(label_lens.get().ravel(),
                                                                   dtype=np.int32),
                                                          np.array(input_lens.get().ravel(),
                                                                   dtype=np.int32),
                                                          nout, self.be.bsz,
                                                          stream_buf)
        self.be.set_scratch_size(scratch_size)
        workspace = self.be.scratch_buffer(scratch_size)

        # map ctc function
        self.ctclib.compute_ctc_loss_gpu.restype = int
        self.ctclib.compute_ctc_loss_gpu.argtypes = [ct.POINTER(ct.c_float),
                                                     ct.POINTER(ct.c_float),
                                                     npct.ndpointer(dtype=np.int32, ndim=1),
                                                     npct.ndpointer(dtype=np.int32, ndim=1),
                                                     npct.ndpointer(dtype=np.int32, ndim=1),
                                                     ct.c_int,
                                                     ct.c_int,
                                                     ct.POINTER(ct.c_float),
                                                     ct.POINTER(ct.c_char),
                                                     ct.c_void_p]

        inputs_buf = ct.cast(int(inputs.gpudata), ct.POINTER(ct.c_float))
        grads_buf = ct.cast(int(grads.gpudata), ct.POINTER(ct.c_float))
        costs_buf = ct.cast(int(costs.gpudata), ct.POINTER(ct.c_float))
        workspace_buf = ct.cast(workspace, ct.POINTER(ct.c_char))

        status = self.ctclib.compute_ctc_loss_gpu(inputs_buf,
                                                  grads_buf,
                                                  np.array(labels.get().ravel(),
                                                           dtype=np.int32),
                                                  np.array(label_lens.get().ravel(),
                                                           dtype=np.int32),
                                                  np.array(input_lens.get().ravel(),
                                                           dtype=np.int32),
                                                  nout,
                                                  self.be.bsz,
                                                  costs_buf,
                                                  workspace_buf,
                                                  stream_buf)

        assert status is 0, "Warp-CTC run failed"
        return

    def be_ctc_cpu(
            self,
            inputs,
            labels,
            grads,
            label_lens,
            input_lens,
            costs,
            nout):
        """
        Calling Warp-CTC
        """

        # Workspace is allocated in ctc_entrypoint.cpp since the CPU backend in neon doesn't have
        # scratch space support
        # Map compute_ctc_loss
        self.ctclib.compute_ctc_loss_cpu.restype = int
        self.ctclib.compute_ctc_loss_cpu.argtypes = [
            npct.ndpointer(dtype=np.float32, ndim=3),
            npct.ndpointer(dtype=np.float32, ndim=3),
            npct.ndpointer(dtype=np.int32, ndim=1),
            npct.ndpointer(dtype=np.int32, ndim=1),
            npct.ndpointer(dtype=np.int32, ndim=1),
            ctypes.c_int,
            ctypes.c_int,
            npct.ndpointer(dtype=np.float32, ndim=1),
            ctypes.c_int]

        num_threads = 8
        _inputs = np.array(inputs.get(), dtype=np.float32)
        _grads = np.array(grads.get(), dtype=np.float32)
        _labels = np.array(labels.get().ravel(), dtype=np.int32)
        _label_lens = np.array(label_lens.get().ravel(), dtype=np.int32)
        _input_lens = np.array(input_lens.get().ravel(), dtype=np.int32)
        _costs = np.array(costs.get().ravel(), dtype=np.float32)
        status = self.ctclib.compute_ctc_loss_cpu(_inputs,
                                                  _grads,
                                                  _labels,
                                                  _label_lens,
                                                  _input_lens,
                                                  nout,
                                                  self.be.bsz,
                                                  _costs,
                                                  num_threads)

        assert status is 0, "Warp-CTC run failed"
        costs[:] = _costs
        grads[:] = _grads
        return

    def bprop(self, y, t):
        # warp-ctc returns grads with shape (T, bsz, nout),
        # so reshape warp-ctc grads to match neon grads
        self.be.copy_transpose(self.grad_warp, self.grad_view, (2, 0, 1))

        return self.grad
