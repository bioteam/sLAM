# sLAM

Code to create a GPT-2-style, decoder-only Small LAnguage Model that can be built using personal computing.

## Installation

```sh
git clone git@github.com:bosborne/sLAM.git
cd sLAM
pip3 install .
```

## Usage

### Gather README files

Requires a GitHub API key. Customize the queries in *get_readmes.py* script.

```sh
python3 sLAM/get_readmes.py 
```

### Create a model

```sh
python3 sLAM/decoder-only-custom-slam.py -d github_readmes/
```

### Query the model

```sh
```
