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
import numpy as np
import sys
from neon.callbacks.callbacks import Callback


class WordErrorRateCallback(Callback):

    def __init__(self, eval_set, decoder, max_s, noise_label=None, epoch_freq=1):
        super(WordErrorRateCallback, self).__init__(epoch_freq=epoch_freq)

        self.eval_set = eval_set
        self.nout = len(decoder.alphabet)
        self.decoder = decoder
        if noise_label is None:
            self.noise_label = ''
        else:
            self.noise_label = noise_label

    def decrypt(self, decoder, message, noise_label):
        msg = decoder.convert_to_string(message)
        return decoder.process_string(msg, remove_repetitions=False
                                      ).replace(noise_label, '')

    def softmax(self, x):
        return (np.reciprocal(np.sum(
                np.exp(x - np.max(x, axis=0)), axis=0)) *
                np.exp(x - np.max(x, axis=0)))

    def dev_to_host(self, dev_tensor):
        if self.be.distribute_data(dev_tensor, "Disabled"):
            revert = True
        else:
            revert = False
        host_tensor = dev_tensor.get()
        if revert:
            self.be.revert_tensor(dev_tensor)
        return host_tensor

    def get_outputs(self, model, inputs):
        outputs = model.fprop(inputs, inference=True)
        return self.softmax(self.dev_to_host(outputs)).reshape(
            (self.nout, -1, self.be.bsz)).transpose((2, 0, 1))

    def get_wer(self, model, dataset, noise_symbol=None):
        if noise_symbol is None:
            noise_symbol = ''
        cer = 0
        batch_count = 1e-10
        for x, y in dataset:
            batch_count += 1
            probs = self.get_outputs(model, x)
            strided_tmax = probs.shape[-1]
            flat_labels = self.dev_to_host(y[0])[0,:]
            tscrpt_lens = self.dev_to_host(y[1])[0, :]
            utt_lens = strided_tmax * self.dev_to_host(y[2])[0, :] / 100
            disp_indx = np.random.randint(self.be.bsz)
            for mu in range(self.be.bsz):
                prediction = self.decoder.decode(probs[mu, :, :utt_lens[mu]])
                start = int(np.sum(tscrpt_lens[:mu]))
                target = flat_labels[start:start + tscrpt_lens[mu]].tolist()
                target = self.decrypt(self.decoder, target, noise_symbol)
                cer += self.decoder.cer(prediction, target) / (1.0 * len(target))

                if mu == disp_indx:
                    disp_proposal = prediction
                    disp_target = target
        return cer / (batch_count * self.be.bsz), disp_proposal, disp_target

    def on_epoch_end(self, callback_data, model, epoch):
        cer, disp_proposal, disp_target = self.get_wer(model, self.eval_set)
        sys.stdout.write('Proposal: ' + disp_proposal + '\n, Target: ' + disp_target)
        sys.stdout.write('\n')
        print("CER (validation) = {}".format(cer))
        sys.stdout.write('\n')
