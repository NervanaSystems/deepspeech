import numpy as np
import theano
import theano.tensor as T
import lasagne

import ctc

num_classes = 5
mbsz = 1
min_len = 12
max_len = 12
n_hidden = 100
grad_clip = 100

input_lens = T.ivector('input_lens')
output = T.ivector('output')
output_lens = T.ivector('output_lens')

l_in = lasagne.layers.InputLayer(shape=(mbsz, max_len, num_classes))

h1f = lasagne.layers.RecurrentLayer(l_in, n_hidden, grad_clipping=grad_clip,
        nonlinearity=lasagne.nonlinearities.rectify)
h1b = lasagne.layers.RecurrentLayer(l_in, n_hidden, grad_clipping=grad_clip,
        nonlinearity=lasagne.nonlinearities.rectify, backwards = True)
h1 = lasagne.layers.ElemwiseSumLayer([h1f, h1b])

h2f = lasagne.layers.RecurrentLayer(h1, n_hidden, grad_clipping=grad_clip,
        nonlinearity=lasagne.nonlinearities.rectify)
h2b = lasagne.layers.RecurrentLayer(h1, n_hidden, grad_clipping=grad_clip,
        nonlinearity=lasagne.nonlinearities.rectify, backwards = True)
h2 = lasagne.layers.ElemwiseSumLayer([h2f, h2b])

h3 = lasagne.layers.RecurrentLayer(h2, num_classes, grad_clipping=grad_clip,
        nonlinearity=lasagne.nonlinearities.linear)
# Turn <minibatch_size, max_length, num_classes> into <max_length, minibatch_size, num_classes>
l_out = lasagne.layers.DimshuffleLayer(h3, (1, 0, 2))

network_output = lasagne.layers.get_output(l_out)

cost = T.mean(ctc.cpu_ctc_th(network_output, input_lens, output, output_lens))
grads = T.grad(cost, wrt=network_output)
all_params = lasagne.layers.get_all_params(l_out)
updates = lasagne.updates.adam(cost, all_params, 0.001)

train = theano.function([l_in.input_var, input_lens, output, output_lens], cost, updates=updates)
predict = theano.function([l_in.input_var], network_output)
get_grad = theano.function([l_in.input_var, input_lens, output, output_lens], grads)

from loader import DataLoader
data_loader = DataLoader(mbsz=mbsz, min_len=min_len, max_len=max_len, num_classes=num_classes)

i = 1
while True:
    i += 1
    print i
    sample = data_loader.sample()
    cost = train(*sample)
    out = predict(sample[0])
    print cost
    print "input", sample[0][0].argmax(1)
    print "prediction", out[:, 0].argmax(1)
    print "expected", sample[2][:sample[3][0]]
    if i == 10000:
        grads = get_grad(*sample)
        import ipdb; ipdb.set_trace()


