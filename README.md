# sLAM

Code to create a GPT-2-style, decoder-only Small LAnguage Model that can be built using personal computing.

## Installation

```sh
git clone git@github.com:bosborne/sLAM.git
cd sLAM
pip3 install .
```

## Usage

### Gather some GitHub README files

Requires a GitHub API key. Customize the queries in the *get_readmes.py* script.

```sh
python3 sLAM/get-readmes.py 
```

### Create a model

```sh
python3 sLAM/make-slam.py -d github_readmes/
```

### Query the model

```sh
```
