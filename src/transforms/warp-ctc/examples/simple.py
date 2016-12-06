from ctc import cpu_ctc_th, cpu_ctc_np
import numpy as np
import theano
import theano.tensor as T

acts = np.array([[[0.1, 0.6, 0.1, 0.1, 0.1]],
                 [[0.1, 0.1, 0.6, 0.1, 0.1]]])

labels = np.array([1, 2])
label_lens = np.array([2])
act_lens = np.array([2])
cost, grads = cpu_ctc_np(acts, act_lens, labels, label_lens)
print "expected cost:", 2.46285844

print "cost (numpy):", cost.sum()
print "grads (numpy):", grads

def create_theano_func():
    acts = T.ftensor3()
    act_lens = T.ivector()
    labels = T.ivector()
    label_lens = T.ivector()
    costs = cpu_ctc_th(acts, act_lens, labels, label_lens)
    cost = T.mean(costs)
    grads = T.grad(cost, acts)
    f = theano.function([acts, act_lens, labels, label_lens], cost, allow_input_downcast=True)
    g = theano.function([acts, act_lens, labels, label_lens], grads, allow_input_downcast=True)
    return f, g

f, g = create_theano_func()
print "cost (theano):", f(acts, act_lens, labels, label_lens).sum()
print "grads (theano)", g(acts, act_lens, labels, label_lens)











