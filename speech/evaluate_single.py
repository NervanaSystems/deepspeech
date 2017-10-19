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

import os
import tempfile

from tqdm import tqdm

from neon.backends import gen_backend
from neon.util.argparser import NeonArgparser, extract_valid_args
from neon.models import Model

from decoder import ArgMaxDecoder
from utils import get_outputs
from data.dataloader import make_inference_loader

# Parse the command line arguments
arg_defaults = {'batch_size': 1}

parser = NeonArgparser(__doc__, default_overrides=arg_defaults)
parser.add_argument("audio_files", nargs="*",
                    help="Audio files to transcribe. They must all be single channel, "
                         "16bit depth with a 16kHz sampling rate")
args = parser.parse_args()

if args.model_file is None:
    raise ValueError("A model file is required for evaluation")

# Setup parameters for argmax decoder
alphabet = "_'ABCDEFGHIJKLMNOPQRSTUVWXYZ "
nout = len(alphabet)
argmax_decoder = ArgMaxDecoder(alphabet, space_index=alphabet.index(" "))

# Initialize our backend
be = gen_backend(**extract_valid_args(args, gen_backend))

# Setup dataloader
eval_manifest = tempfile.mktemp(prefix="manifest_", suffix=".tsv")
with open(eval_manifest, "w") as fh:
    fh.write("@FILE\n")
    for audio_file in args.audio_files:
        if not os.path.isfile(audio_file):
            raise IOError("Audio file does not exist: {}".format(audio_file))
        fh.write("{}\n".format(audio_file))

# Setup required dataloader parameters
nbands = 13
max_utt_len = 30
max_tscrpt_len = 1300
eval_set = make_inference_loader(eval_manifest, nbands, max_utt_len, backend_obj=be)

# Load the model
model = Model(args.model_file)
if not model.initialized:
    model.initialize(eval_set)

# Loop through and process audio
index = 0
for audio, audio_len in tqdm(eval_set, unit="files" if be.bsz == 1 else "batches",
                             total=eval_set.nbatches):
    output = get_outputs(model, model.be, audio, nout)
    strided_tmax = output.shape[-1]
    utt_lens = strided_tmax * audio_len.get().ravel() / 100
    for ii in range(be.bsz):
        transcript = argmax_decoder.decode(output[ii, :, :int(utt_lens[ii])])
        tqdm.write("File: {}\nTranscript: {}".format(args.audio_files[index], transcript))
        index += 1
