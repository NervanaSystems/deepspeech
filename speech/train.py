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
"""
Train ds2-style speech model on Librispeech
"""

import os
import numpy as np

from neon.backends import gen_backend
from neon.callbacks.callbacks import Callbacks
from neon.initializers import GlorotUniform, Constant, Gaussian
from neon.layers import Conv, GeneralizedCost, Affine, DeepBiRNN
from neon.models import Model
from neon.transforms import Rectlin, Identity, Rectlinclip
from neon.optimizers import GradientDescentMomentum
from neon.util.argparser import NeonArgparser, extract_valid_args

from ctc import CTC
from decoder import ArgMaxDecoder
from sample_proposals_callback import WordErrorRateCallback
from data.dataloader import make_loader

# Parse the command line arguments
arg_defaults = {'batch_size': 32}

parser = NeonArgparser(__doc__, default_overrides=arg_defaults)
parser.add_argument('--nfilters', type=int,
                    help='no. of conv filters', default=1152)
parser.add_argument('--filter_width', type=int,
                    help='width of conv filter', default=11)
parser.add_argument('--str_w', type=int, help='stride in time', default=3)
parser.add_argument('--depth', type=int, help='rnn depth', default=9)
parser.add_argument('--hidden_size', type=int,
                    help='affine/rnn hidden units', default=1152)
parser.add_argument('--lr', type=float,
                    help='learning rate', default=2e-5)
parser.add_argument('--momentum', type=float,
                    help='momentum', default=0.99)
args = parser.parse_args()

# Setup model hyperparameters
# Convolution layer hyperparameters
nfilters = args.nfilters  # Number of convolutional filters
filter_width = args.filter_width  # Width of convolutional filters
str_w = args.str_w  # Convolutional filter stride

# RNN hyperparameters
depth = args.depth  # Number of BiRNN layers
hidden_size = args.hidden_size # Number of units in each BiRNN layer

# Optimization hyperparameters
learning_rate = args.lr
momentum = args.momentum
gradient_clip_norm = 400

# Setup parameters for argmax decoder
alphabet = "_'ABCDEFGHIJKLMNOPQRSTUVWXYZ "
nout = len(alphabet)
argmax_decoder = ArgMaxDecoder(alphabet, space_index=alphabet.index(" "))

# Initialize our backend
be = gen_backend(**extract_valid_args(args, gen_backend))

# Setup dataloader
nbands = 13
max_tscrpt_len = 1300
max_utt_len = 30

train_manifest = args.manifest['train']
if not os.path.exists(train_manifest):
    raise RuntimeError(
        "training manifest file {} not found".format(train_manifest))
dev_manifest = args.manifest['val']
if not os.path.exists(dev_manifest):
    raise RuntimeError(
        "validation manifest file {} not found".format(dev_manifest))

train = make_loader(train_manifest, alphabet, nbands, max_tscrpt_len, max_utt_len, backend_obj=be)
dev = make_loader(dev_manifest, alphabet, nbands, max_tscrpt_len, max_utt_len, backend_obj=be)

# Setup the layers of the DNN
# Softmax is performed in warp-ctc, so we use an Identity activation in the
# final layer.
gauss = Gaussian(scale=0.01)
glorot = GlorotUniform()
layers = [
    Conv(
        (nbands,
         filter_width,
         nfilters),
        init=gauss,
        bias=Constant(0),
        activation=Rectlin(),
        padding=dict(
            pad_h=0,
            pad_w=5),
        strides=dict(
            str_h=1,
            str_w=str_w)),
    DeepBiRNN(
        hidden_size,
        init=glorot,
        activation=Rectlinclip(),
        batch_norm=True,
        reset_cells=True,
        depth=depth),
    Affine(
        hidden_size,
        init=glorot,
        activation=Rectlinclip()),
    Affine(
        nout=nout,
        init=glorot,
        activation=Identity())]

model = Model(layers=layers)

opt = GradientDescentMomentum(learning_rate, momentum,
                              gradient_clip_norm=gradient_clip_norm,
                              stochastic_round=False,
                              nesterov=True)
callbacks = Callbacks(model, eval_set=dev, **args.callback_args)

# Print validation set word error rate at the end of every epoch
pcb = WordErrorRateCallback(dev, argmax_decoder, max_tscrpt_len, epoch_freq=1)
callbacks.add_callback(pcb)

cost = GeneralizedCost(costfunc=CTC(max_tscrpt_len, nout=nout))

# Fit the model
model.fit(train, optimizer=opt, num_epochs=args.epochs,
          cost=cost, callbacks=callbacks)
