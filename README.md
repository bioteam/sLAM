# sLAM

Demonstration code to create a GPT-2-style, decoder-only, generative small LAnguage Model that can be built using personal computing.

This is not for production. You can use this code to learn about generative language models, preprocessing, and training hyperparameters.

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

The code uses *wikitext-2-v1* or *cs_news* from Hugging Face as training text, e.g. *-d cs_news*.

### Build a model

Download and clean training data from *cs_news*, create a model, train the model with 1% of the cleaned chunks for 3 epochs, be verbose, and try the given prompt:
  
```sh
python3 sLAM/make-slam.py -d cs_news -p "This is a test" -t 1 -v --epochs 3
```

This creates a Keras model and a saved (serialized) tokenizer with the same name, and a histogram of sentence lengths. for example:

```sh
-rw-r--r--   332M Apr  1 05:09 04-01-2025-05-09-04.keras
-rw-r--r--    58K Apr  1 05:09 04-01-2025-05-09-04.pickle
-rw-r--r--    19K Mar 31 16:04 sentence_length_distribution.png
```

One epoch takes about 4.5 hours on a Mac M1 laptop (32 GB RAM).

### Generate using an existing model

Supply the name of the model and the serialized tokenizer, and a prompt:

```sh
python3 sLAM/generate.py -n 04-01-2025-05-09-04 -p "this is a test"
```

Some results from a model trained for 3 epochs with ~100K tokens, generated at different temperatures:

* This is a test the contract gave a revenue of up to 300 million in the course five years
* This is a test the hurricane began to turn more northwestward in response to a high pressure system weakening to its north
* this is a test right when reubens begins to snap danny out of hypnosis crush
* this is a test it pond and the route wanted of what is today on july 21 humor and is according to burn for 21 although there

Some of these examples suggest that the model is "memorizing" rather than generating novel text. It's likely that the training data set is too small, and/or that overfitting may be occuring.

When the model is trained on ~1M tokens the generated text starts to look more syntactically correct, with less memorization:

* This is a test however information shannon argued that it is not feasible for any computer to actually do this
* This is a test the fourth attacked title was the first ship in the play s performance of the winner in the uk and was released as part of the season
* This is a test a video posted to youtube called code of conduct outlined twenty two rules to follow when protesting and kakapo from a urged

## To Do

* Handle end-of-sentence (EOS) correctly.
* Implement *mask_zero=True* in the embedding layer so that padding in the prompt is ignored during generation.

## Operating Notes

* Reducing the size of the input text with *-t* can eliminate OOM errors on the RTX 5000.
* Reducing the size of the embedding with *--d_model* significantly reduces training time, e.g. from 20ms/step to 5ms/step
* *context_size* should be related to chunk size, which averages about 38 tokens for the cc_news data
* Reducing *context_size* reduces time per step but also reduces accuracy, so more training may be required

### Using Tensorflow from a container at TACC

Run an interactive job using `srun` or `idev` to download a container made by NVIDIA.

```sh
srun -N 1 -n 10 -p rtx-dev -t 60:00 --pty bash
module load tacc-apptainer
apptainer pull docker://tensorflow/tensorflow:2.17.0-gpu
```

Once the container is downloaded you can run it with `singularity`.

```sh
singularity shell --nv tensorflow_2.17.0-gpu.sif
```
