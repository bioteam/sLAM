# sLAM

Demonstration code to create a GPT-2-style, decoder-only, generative small LAnguage Model that can be built using personal computing. This is not for production. You can use this code to learn about generative language models, preprocessing, and training hyperparameters.

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
usage: make-slam.py [-h] [-i INPUT_DIR] [-d] [-t TEXT_PERCENTAGE] [-n NAME]
                    [--min_sentence_len MIN_SENTENCE_LEN] [--temperature TEMPERATURE] [--epochs EPOCHS] -p
                    PROMPT [-v]

options:
  -h, --help            show this help message and exit
  -i INPUT_DIR, --input_dir INPUT_DIR
                        Specify a directory with text files
  -d, --download        Do a text download from Hugging Face
  -t TEXT_PERCENTAGE, --text_percentage TEXT_PERCENTAGE
                        Percentage of input text used to make dataset
  -n NAME, --name NAME  Name used to save files, default is timestamp of completion
  --min_sentence_len MIN_SENTENCE_LEN
                        Mininum sentence length used in training
  --temperature TEMPERATURE
                        Temperature used for generation
  --epochs EPOCHS       Number of epochs
  -p PROMPT, --prompt PROMPT
                        Prompt
  -v, --verbose         Verbose

```

The code uses *wikitext-2-v1* from Hugging Face as training text if *-d* is specified.

### Build a model

Download and clean *wikitext-2-v1*, create a model, train the model with 1% of the cleaned *wikitext-2-v1* sentences for 3 epochs, be verbose, and use the given prompt:

```sh
python3 sLAM/make-slam.py -d -p "This is a test" -t 1 -v --epochs 3
```

With these hyperparameters and inputs 1 epoch takes about 4.5 hours on a Mac M1 laptop (32 GB RAM).

Some results from 3 epochs:

* This is a test the contract gave a revenue of up to 300 million in the course five years
* This is a test the hurricane began to turn more northwestward in response to a high pressure system weakening to its north

These examples suggest that the model is "memorizing" rather than generating novel text. It's likely that the training data set is too small, and that overfitting may be occuring.

### Generate using an exising model

```sh
python3 sLAM/generate.py -n 04-01-2025-05-09-04 -p "this is a test"
```

## To Do

* Add a validation dataset to test validation loss and other metrics.
* Use validation loss as a metric for *early_stopping*
* Experiment with larger input texts
* Handle end-of-sentence (EOS) correctly.
* Implement *mask_zero=True* in the embedding layer.
* Optional: Add subword tokenization capability.
