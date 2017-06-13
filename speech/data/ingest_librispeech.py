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
import os
import logging
import glob
import fnmatch
import json

from neon.data.aeon_shim import AeonDataLoader
from neon.data.dataloaderadapter import DataLoaderAdapter
from neon.data.dataloader_transformers import TypeCast, Retuple

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def write_manifest(output_file, *filenames):
    """ Writes out a manifest file from a series of lists of filenames
    """

    with open(output_file, "w") as fid:
        fid.write("@FILE\tFILE\n");
	for line in zip(*filenames):
            fid.write("\t".join(line) + "\n")

    return True

def main(input_directory, transcript_directory, manifest_file):
    """ Finds all .flac files recursively in input_directory, then extracts the 
    transcript from the nearby .trans.txt file and stores it in
    transcript_directory. Writes a manifest file referring to each .flac file
    and its paired transcript.
    
    Arguments:
        input_directory (string): Path to librispeech directory
        transcript_directory (string): Path to directory in which to write
            individual transcript files.
        manifest_file (string): Path to manifest file to output.
    """

    def librispeech_flac_filename(filestr):
        parts = filestr.split("-")

        return os.path.join(input_directory, parts[0], parts[1],
                            "{}.flac".format(filestr))

    if not os.path.isdir(input_directory):
        raise IOError("Data directory does not exist! {}".format(input_directory))

    if not os.path.exists(transcript_directory):
        os.makedirs(transcript_directory)

    transcript_files = glob.glob(os.path.join(input_directory, '*/*/*.txt'))
    if len(transcript_files) == 0:
        logger.error("No .txt files were found in {}".format(input_directory))
        return

    logger.info("Beginning audio conversions")
    audio_files = list()
    txt_files = list()
    for ii, tfile in enumerate(transcript_files):
        # transcript file specifies transcript and flac filename for all librispeech files
        logger.info("Converting audio corresponding to transcript "
                    "{} of {}".format(ii, len(transcript_files)))
        with open(tfile, "r") as fid:
            lines = fid.readlines()

        for line in lines:
            filestr, transcript = line.split(" ", 1)
            try:
                flac_file = librispeech_flac_filename(filestr)
            except IndexError: # filestr is not the format we are expecting
                print("filestr of unexpected formatting: {}".format(filestr))
                print("error in {}".format(tfile))
                continue
            txt_file = os.path.join(transcript_directory,
                                    "{}.txt".format(filestr))
            
            # Write out short transcript file
            with open(txt_file, "w") as fid:
                fid.write(transcript.strip())

            # Add to output lists to be written to manifest
            audio_files.append(flac_file)
            txt_files.append(txt_file)

    logger.info("Writing manifest file to {}".format(manifest_file))
    return write_manifest(manifest_file, audio_files, txt_files)

def common_config(manifest_file, batch_size, alphabet, nbands, max_tscrpt_len):

    audio_config = {"type": "audio",
                            "sample_freq_hz": 16000,
                            "max_duration": "30 seconds",
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

    dl = DataLoaderAdapter(dl)
    dl = TypeCast(dl, index=0, dtype=np.float32)
    dl = Retuple(dl, data=(0,), target=(1, 2, 3))
    return dl

def make_loader(manifest_file, alphabet, nbands, max_tscrpt_len, backend_obj):
    aeon_config = common_config(manifest_file, backend_obj.bsz, alphabet, nbands, max_tscrpt_len)
    return wrap_dataloader(AeonDataLoader(json.dumps(aeon_config)))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_directory",
                        help="Directory containing librispeech flac files")
    parser.add_argument("transcript_directory",
                        help="Directory to write transcript .txt files")
    parser.add_argument("manifest_file",
                        help="Output file that specifies the filename for each"
                        " output audio and transcript")

    args = parser.parse_args()
    main(args.input_directory,
         args.transcript_directory,
         args.manifest_file)
