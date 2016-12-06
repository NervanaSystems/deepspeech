import platform
import os
import numpy as np
import numpy.ctypeslib as npct
import ctypes
import ctypes.util

import theano
import theano.tensor as T
from theano.gradient import grad_undefined

if platform.system() == "Darwin":
    ext = "dylib"
elif platform.system() == "Linux":
    ext = "so"
else:
    raise Exception("Unsupported platform: {}".format(platform.system()))
libwarpctc = npct.load_library(os.path.join(os.path.dirname(__file__), "../build/libwarpctc.{}".format(ext)), "")

libwarpctc.cpu_ctc.restype = None
libwarpctc.cpu_ctc.argtypes = [
        npct.ndpointer(dtype=np.float32, ndim=3),
        npct.ndpointer(dtype=np.float32, ndim=3),
        npct.ndpointer(dtype=np.int32, ndim=1),
        npct.ndpointer(dtype=np.int32, ndim=1),
        npct.ndpointer(dtype=np.int32, ndim=1),
        ctypes.c_int,
        ctypes.c_int,
        npct.ndpointer(dtype=np.float32, ndim=1),
        ctypes.c_int]

def cpu_ctc_np(acts, act_lens, labels, label_lens):
    """
    acts: 3-d numpy float array, same as c++ bindings
    act_lens: 1-d int array of input length of each example
    labels: list of 1-d int array for each example in minibatch
    label_lens: 1-d int array of label length of each example
    """
    # make sure correct types
    acts = np.array(acts, dtype=np.float32)
    act_lens = np.array(act_lens, dtype=np.int32)
    labels = np.array(labels, dtype=np.int32)
    label_lens = np.array(label_lens, dtype=np.int32)

    # C needs sizes
    alphabet_size = acts.shape[2]
    minibatch = acts.shape[1]

    # create return variables
    grads = np.zeros_like(acts, dtype=np.float32)
    cost = np.zeros((minibatch,), dtype=np.float32)

    # compute
    libwarpctc.cpu_ctc(acts, grads, labels, label_lens, act_lens, alphabet_size, minibatch, cost, 1)
    return cost, grads

class CPUCTCGrad(theano.Op):
    # Properties attribute
    __props__ = ()

    def make_node(self, *inputs):
        inputs = map(theano.tensor.as_tensor_variable, inputs)
        # add checks here for types and numdims of all inputs
        return theano.Apply(self, inputs, [T.ftensor3()])

    def perform(self, node, inputs, outputs):
        inputs[0] = inputs[0].astype(np.float32)
        inputs[1] = inputs[1].astype(np.int32)
        inputs[2] = inputs[2].astype(np.int32)
        inputs[3] = inputs[3].astype(np.int32)
        cost, gradients = cpu_ctc_np(*inputs)
        outputs[0][0] = gradients

class CPUCTC(theano.Op):
    # Properties attribute
    __props__ = ()

    def make_node(self, *inputs):
        inputs = map(theano.tensor.as_tensor_variable, inputs)
        # add checks here for types and numdims of all inputs
        return theano.Apply(self, inputs, [T.fvector()])

    def perform(self, node, inputs, outputs):
        inputs[0] = inputs[0].astype(np.float32)
        inputs[1] = inputs[1].astype(np.int32)
        inputs[2] = inputs[2].astype(np.int32)
        inputs[3] = inputs[3].astype(np.int32)
        cost, gradients = cpu_ctc_np(*inputs)
        outputs[0][0] = cost

    def grad(self, inputs, output_grads):
        gradients = CPUCTCGrad()(*inputs)
        return [gradients,
                grad_undefined(self, 1, inputs[1]),
                grad_undefined(self, 2, inputs[2]),
                grad_undefined(self, 3, inputs[3])]

cpu_ctc_th = CPUCTC()


