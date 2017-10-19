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

import sys
import numpy as np
from tqdm import tqdm


def softmax(x):
    return (np.reciprocal(np.sum(
            np.exp(x - np.max(x, axis=0)), axis=0)) *
            np.exp(x - np.max(x, axis=0)))

def get_outputs(model, be, inputs, nout):
    outputs = model.fprop(inputs, inference=False)
    return softmax(outputs.get()).reshape(
        (nout, -1, be.bsz)).transpose((2, 0, 1))

def eval_model(model, dataset, nout, inference=False):
    if inference is True:
        return [(get_outputs(model, model.be, x, nout), x_len) for x, x_len in dataset]
    else:
        return [((get_outputs(model, model.be, x, nout), y[0]), y[2]) for (x, y) in dataset]

def decrypt(decoder, message):
    msg = decoder.convert_to_string(message)
    return decoder.process_string(msg, remove_repetitions=False)

def get_wer(model, be, dataset, decoder, nout, use_wer=False, print_examples=False):
    wer = 0
    batchcount = 0
    predictions = list()
    targets = list()
    nbatches = dataset.nbatches

    if not model.initialized:
        model.initialize(dataset)
    
    progress_bar = tqdm(dataset, total=nbatches, unit="batches")
    for x, y in progress_bar:
        probs = get_outputs(model, be, x, nout)
        strided_tmax = probs.shape[-1]
        flat_labels = y[0].get().ravel()
        tscrpt_lens = y[1].get().ravel()
        utt_lens = strided_tmax * y[2].get().ravel() / 100
        for mu in range(be.bsz):
            prediction = decoder.decode(probs[mu, :, :int(utt_lens[mu])])
            start = int(np.sum(tscrpt_lens[:mu]))
            target = flat_labels[start:start + tscrpt_lens[mu]].tolist()
            target = decrypt(decoder, target)
            predictions.append(prediction)
            targets.append(target)
            if not use_wer:
                wer += decoder.cer(prediction, target) / float(len(target))
            else:
                wer += decoder.wer(prediction, target) / \
                        float(len(target.split()))
        
        if use_wer:
            progress_bar.set_description("WER: {}".format(wer / len(predictions)))
        else:
            progress_bar.set_description("CER: {}".format(wer / len(predictions)))
        if print_examples is True:
            progress_bar.write("Transcribed: {}".format(predictions[-1]))
            progress_bar.write("Target:      {}".format(targets[-1]))

    results = zip(predictions, targets)
    nsamples = len(predictions)
    return wer / nsamples, nsamples , results
