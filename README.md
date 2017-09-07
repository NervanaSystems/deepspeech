# Implementation of Deep Speech 2 in neon

This repository contains an implementation of Baidu SVAIL's [deep speech 2](https://arxiv.org/abs/1512.02595) model in neon. Much of the model is readily available in mainline neon, but to support the CTC cost function we have included a neon-compatible wrapper for Baidu's [Warp-CTC].
  
The deep speech 2 model is very computationally intensive. Even at near perfect GPU utilization, the model can take up to 1 week to train on large enough datasets to see respectable performance. Thus, we suggest the user keep this in mind when exploring this repo. We have used this code to train models on both the Wall Street Journal (81 hours) and Librispeech (1000 hours) datasets. The WSJ dataset is only available through the LDC, but Librispeech can be freely acquired from <http://www.openslr.org/12/>.
 
The model presented here uses a basic argmax-based decoder, simply choosing the most probable character in each frame and then collapsing the resulting output string according to CTC's rules (first remove repeat characters, then remove blank characters). With this decoding, you might expect outputs like this when trained on WSJ:

| Ground truth                    | Model output                      |
|---------------------------------|-----------------------------------|
| united presidential is a life insurance company | younited presidentiol is a lefe in surance company |
| that was certainly true last week | that was sertainly true last week |
| we're not ready to say we're in technical default a spokesman said | we're now ready to say we're intechnical default a spokesman said | 

Or like this when trained on Librispeech (see "Decoding and evaluating a trained model"):

| Ground truth                    | Model output                      |
|---------------------------------|-----------------------------------|
| this had some effect in calming him | this had some offectind calming him |
| he went in and examined his letters but there was nothing from carrie | he went in an examined his letters but there was nothing from carry |
| the design was different but the thing was clearly the same | the design was differampat that thing was clarly the same |

## Getting Started
1. Make sure you have [neon](https://github.com/NervanaSystems/neon) installed.  
    a. This repo also uses the [aeon](https://github.com/NervanaSystems/aeon) dataloader. If you chose not to install it with neon, you must install it manually.

2. Within a neon virtualenv, run ```pip install python-levenshtein```.

3. Clone this repository: ```git clone https://github.com/NervanaSystems/deepspeech.git && cd deepspeech```

4. Run ```make``` to build warp-ctc.

## Training a model
### 1. Prepare a manifest file for your dataset.
The details on how to go about doing this are determined by the specifics of the dataset. 


#### Example: Librispeech recipe
We have provided a recipe for ingesting Librispeech data in data/ingest_librispeech.py. Note that Librispeech provides distinct datasets for training and validation, so you will need to ingest each dataset separately. Additionally, we have to get around the quirky way that the Librispeech data is distributed, so after "unpacking" the archives, we should re-pack them in a consistent manner.

To be more precise, Librispeech data is distributed in zipped tar files, e.g. `train-clean-100.tar.gz` for training and `dev-clean.tar.gz` for validation. Upon unpacking, each archive creates a directory named LibriSpeech, which means you cannot unpack both files together in the same directory. To get around this, we suggest the following. 
```
mkdir librispeech 
cd librispeech
wget http://www.openslr.org/resources/12/train-clean-100.tar.gz
wget http://www.openslr.org/resources/12/dev-clean.tar.gz
tar xvzf dev-clean.tar.gz LibriSpeech/dev-clean  --strip-components=1
tar xvzf train-clean-100.tar.gz LibriSpeech/train-clean-100  --strip-components=1
```
If you follow the above prescription, you will have the training data as a subdirectory `librispeech/train-clean-100` 
and  the validation data in a subdirectory `librispeech/dev-clean`. To ingest the data, you would then run 
```
python data/ingest_librispeech.py <absolute path to train-clean-100 directory> <absolute path to directory to write transcripts to> <absolute path to where to write training manifest to>
```

For example, if the absolute path to the train-clean-100 directory is ``/usr/local/data/librispeech/train-clean-100``, you would issue  
```
python data/ingest_librispeech.py  /usr/local/data/librispeech/train-clean-100  /usr/local/data/librispeech/train-clean-100/transcripts_dir  /usr/local/data/librispeech/train-clean-100/train-manifest.csv
```
which would create a training manifest file named train-manifest.csv. Similarly, if the absolute path to the dev-clean directory is ``/usr/local/data/librispeech/dev-clean``, you would issue  

```
python data/ingest_librispeech.py  /usr/local/data/librispeech/dev-clean  /usr/local/data/librispeech/dev-clean/transcripts_dir  /usr/local/data/librispeech/train-clean-100/val-manifest.csv
```

To train on the full 1000 hours, you should execute the same commands for the 360 hour and 540 hour training datasets as well. The manifest files can then be concatenated with a simple:
`cat /path/to/100_hour_manifest.csv /path/to/360_hour_manifest.csv /path/to/540_hour_manifest.csv > /path/to/1000_hour_manifest.csv`. 


### 2a. Train a new model

```
python train.py --manifest train:<training manifest> --manifest val:<validation manifest> -e <num_epochs> -z <batch_size> -s </path/to/model_output.pkl> [-b <backend>] 
```

where `<training manifest>` is the path to the training manifest file produced in the ingest step above (e.g. ``/usr/local/data/librispeech/train-clean-100/train-manifest.csv`` in the example above) and `<validation manifest>` is the path to the validation manifest file.
 
### 2b. Continue training a previous model
If you have a previously trained model, you can resume training by passing the `--model_file </path/to/pre-trained_model>` argument to `train.py`. For example, you could continue training a pre-trained model from our [Model Zoo](https://github.com/NervanaSystems/ModelZoo). This particular model was trained using 1000 hours of speech data from the [Librispeech corpus](<http://www.openslr.org/12/>). The model was trained for 16 epochs after attaining a character error rate (CER) of 14% without using a language model. You could continue training it for, say, an additional 4 epochs, by calling 
```
python train.py --manifest train:<training manifest> --manifest val:<validation manifest> -e20  -z <batch_size> -s </path/to/model_output.prm> --model_file </path/to/pre-trained_model> [-b <backend>] 
```
which will save a new model to `model_output.prm`. 

## Decoding and evaluating a trained model
Once you have a trained model, you can easily evaluate its performance on any given dataset. Simply create a manifest file and then call:
 ```
 python evaluate.py --manifest val:/path/to/manifest.csv --model_file /path/to/saved_model.prm
 ```
replacing the file paths as needed. This will print out character error rates by default. To print word error rates, include the argument `--use_wer`.

For example, you could evaluate our pre-trained model from our [Model Zoo](https://github.com/NervanaSystems/ModelZoo). To evaluate the pre-trained model, you would first need to 

1. download some test data from the Librispeech ASR corpus and prepare a manifest file for the dataset following the prescription provided above.  

2. download the pre-trained model from our [Model Zoo](https://s3-us-west-1.amazonaws.com/nervana-modelzoo/Deep_Speech/Librispeech/librispeech_16_epochs.prm)

Once you have the pre-trained model and the manifest file for the test data, you would evaluate the model by calling `evaluate.py` as indicated above.

If you would like to inspect the transcripts produced by the trained model, you can include the argument `--inference_file <name_of_file_to_save_results_to.pkl>` which dumps out the model transcripts together with the corresponding "ground truth" transcripts to a pickle file. 


[Warp-CTC]: https://github.com/baidu-research/warp-ctc
