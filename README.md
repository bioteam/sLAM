# sLAM

Demonstration code to create a GPT-2-style, decoder-only Small LAnguage Model that can be built using personal computing. A utility script is included that downloads GitHub READMEs for input.

## Installation

```sh
git clone git@github.com:bosborne/sLAM.git
cd sLAM
pip3 install .
```

## Usage

### Gather some GitHub README files

Use READMEs as input or use your own text documents.

Gathering GitHub READMEs requires a GitHub API key: Settings -> Developer Settings -> Personal access tokens -> Tokens (classic).

Then:

```sh
export GITHUB_API_KEY=abcdefghijklmnopqrstuvwxyz
```

Customize the queries in the *get-readmes.py* script.

```sh
python3 sLAM/get-readmes.py 
```

Or use your own text documents collected in one directory.

### Create a model

Specify the documents directory and make a model.

```sh
python3 sLAM/make-slam.py -d github_readmes/
```

### Query the model

```sh
```
