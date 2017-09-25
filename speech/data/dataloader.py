#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2017 Nervana Systems Inc.
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
import numpy as np
from neon.data.aeon_shim import AeonDataLoader
from neon.data.dataloader_transformers import TypeCast, Retuple


def common_config(manifest_file, batch_size, alphabet, nbands, max_tscrpt_len, max_utt_len):

    audio_config = {"type": "audio",
                    "sample_freq_hz": 16000,
                    "max_duration": "{} seconds".format(max_utt_len),
                    "frame_length": "25 milliseconds",
                    "frame_stride": "10 milliseconds",
                    "feature_type": "mfsc",
                    "emit_length": True,
                    "num_filters": nbands}

    transcription_config = {"type": "char_map",
                            "alphabet": alphabet,
                            "emit_length": True,
                            "max_length": max_tscrpt_len}

    return {'manifest_filename': manifest_file,
            'manifest_root': os.path.dirname(manifest_file),
            'batch_size': batch_size,
            'block_size': batch_size,
            'etl': [audio_config, transcription_config]}


def wrap_dataloader(dl):
    """ Data is loaded from Aeon as a 4-tuple. We need to cast the audio
    (index 0) from int8 to float32 and repack the data into (audio, 3-tuple).
    """

    dl = TypeCast(dl, index=0, dtype=np.float32)
    dl = Retuple(dl, data=(0,), target=(2, 3, 1))
    return dl


def make_loader(manifest_file, alphabet, nbands, max_tscrpt_len, max_utt_len, backend_obj):
    aeon_config = common_config(manifest_file, backend_obj.bsz, alphabet, nbands, max_tscrpt_len,
                                max_utt_len)
    return wrap_dataloader(AeonDataLoader(aeon_config))
