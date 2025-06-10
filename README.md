# sLAM

Demonstration code to create a GPT-2-style, decoder-only, generative small LAnguage Model that can be built using personal computing.

This is not for production. You can use this code to learn about generative language models, preprocessing, training, and model hyperparameters.

## Installation

```sh
git clone git@github.com:bioteam/sLAM.git
cd sLAM
pip3 install .
```

Complete the installation:

```sh
>python3
>>> import nltk
>>> nltk.download('punkt_tab')
```

*nltk* is used for sentence tokenization.

## Usage

```sh
> python3 sLAM/make-slam.py -h
usage: make-slam.py [-h] [-t TEXT_PERCENTAGE] [--context_size CONTEXT_SIZE] [-n NAME] [--temperature TEMPERATURE]
                    [--epochs EPOCHS] [--d_model D_MODEL] [-d DOWNLOAD] [--num_rows NUM_ROWS] [--use_mlflow] 
                    -p PROMPT [-v]

options:
  -h, --help            show this help message and exit
  -t TEXT_PERCENTAGE, --text_percentage TEXT_PERCENTAGE
                        Percentage of download used to make dataset
  -m MIN_SENTENCE_LEN, --min_sentence_len MIN_SENTENCE_LEN
                        Percentage of input text used to make dataset
  -n NAME, --name NAME  Name used to save files, default is timestamp of start time
  --temperature TEMPERATURE
                        Temperature used for generation
  --epochs EPOCHS       Number of epochs
  --d_model D_MODEL     Number of epochs
  --context_size CONTEXT_SIZE     
                        Context size
  -d DOWNLOAD, --download DOWNLOAD
                        Dataset to download. Default is cc_news.
  --num_rows NUM_ROWS   Number of rows to download from cc_news
  --use_mlflow USE_MLFLOW   
                        Use MLFlow for model tracking
  -p PROMPT, --prompt PROMPT
                        Prompt
  -v, --verbose         Verbose
```

The code uses *wikitext-2-v1* or *cs_news* from Hugging Face as training text, e.g. *-d cs_news*.

### Build a model

Download and clean training data from *cs_news*, create a model, train the model the cleaned chunks for 3 epochs, be verbose, and try the given prompt:

```sh
python3 sLAM/make-slam.py -d cs_news --num_rows 500 -v --epochs 3 -p "This is a test"
```

This creates a Keras model (~1M tokens) and a saved (serialized) tokenizer with the same name, and a histogram of sentence lengths. for example:

```sh
-rw-r--r--   332M Apr  1 05:09 04-01-2025-05-09-04.keras
-rw-r--r--    58K Apr  1 05:09 04-01-2025-05-09-04.pickle
-rw-r--r--    19K Mar 31 16:04 sentence_length_distribution.png
```

One epoch takes about ~1 hour on a Mac M1 laptop (32 GB RAM) with the command above.

### Generate using an existing model

Supply the name of the model and the serialized tokenizer, and a prompt:

```sh
python3 sLAM/generate.py -n 04-01-2025-05-09-04 -p "This is a test"
This is a test if your favorite software is the news service for the bottom of the increasing equipment market is actually plans for their concerns and the narrative of the same time i think it was the course of the technology is that the 5th us and i think what we are the most youre doing it we do to do that you want what to avoid the first amendment and other candidates are not just as the most.
```

### Library and package versions

One of the challenges in writing and running Deep Learning code is how many components there are, and how quickly new versions of these components appear. To get all your component versions aligned start with your computer, which may be a GPU. For example, if it's NVIDIA, what is the recommended version of CUDA? From that version find the recommended version of Tensorflow or Pytorch. Then for that package version what version of Python. An example set of versions, working with an older NVIDIA GPU:

RTX 5000 + CUDA 11.8 + Tensorflow 2.12 + Python 3.8

Then the Python dependencies will follow from the Python version.

*Getting these versions aligned is critical*, because if the versions are out of alignment you may get errors of various kinds that do not reference versions but are more generic and difficult to debug, like out-of-memory errors.

#### Using Tensorflow from a container

Containers may be available that package the right versions of CUDA with some framework. For example,
at TACC you can download a container made by NVIDIA which supplies CUDA, Python, and Tensorflow:

```sh
srun -N 1 -n 10 -p rtx-dev -t 60:00 --pty bash
module load tacc-apptainer
apptainer pull docker://tensorflow/tensorflow:2.12.0-gpu
```

Once the container is downloaded you can run it with `singularity`.

```sh
singularity shell --nv tensorflow_2.12.0-gpu.sif
```
