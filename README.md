# sLAM

Demonstration code to create a GPT-2-style, decoder-only small LAnguage Model that can be built using personal computing.

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

## Usage

```sh
usage: make-slam.py [-h] [-i INPUT_DIR] [-d] [-t TEXT_PERCENTAGE] [-n NAME] [--temperature TEMPERATURE] -p PROMPT [-v]

options:
  -h, --help            show this help message and exit
  -i INPUT_DIR, --input_dir INPUT_DIR
                        Specify a directory with text files
  -d, --download        Do a text download from Hugging Face
  -t TEXT_PERCENTAGE, --text_percentage TEXT_PERCENTAGE
                        Percentage of input text used to make dataset
  -n NAME, --name NAME  Name used to save files, default is timestamp of completion
  --temperature TEMPERATURE
                        Temperature
  -p PROMPT, --prompt PROMPT
                        Prompt
  -v, --verbose         Verbose
```

The code uses *wikitext-2-v1* from Hugging Face as training text if *-d* is specified.

### Create a model and generate

For example, create a model, train with *wikitext-2-v1*, be verbose, and use a prompt:

```sh
python3 sLAM/make-slam.py -p "I am testing a language model" -d -v
```
