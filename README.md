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

The code uses *wikitext-2-v1* from Hugging Face as training text.

### Create a model and generate

Create a model and supply a prompt:

```sh
python3 sLAM/make-slam.py -p "I am testing a language model" -d -v
```

### Query the model

```sh
```
