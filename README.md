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

The code uses *cs_news* (the default) or *wikitext-2-v1* from Hugging Face as training text.

### Build a model

Download and clean training data from *cs_news*, tokenize it into large chunks, create a model, train the model using context-window-sized slices for 3 epochs, be verbose, and try the given prompt:

```sh
python3 sLAM/make-slam.py --num_rows 500 -v --epochs 3 -p "This is a test"
```

This creates a Keras model (~1M input tokens) and a saved (serialized) tokenizer with the same name, and a histogram of sentence lengths. for example:

```sh
-rw-r--r--   332M Apr  1 05:09 04-01-2025-05-09-04.keras
-rw-r--r--    58K Apr  1 05:09 04-01-2025-05-09-04.pickle
-rw-r--r--    19K Mar 31 16:04 sentence_length_distribution.png
```

One epoch takes about ~1 hour on a Mac M1 laptop (32 GB RAM) with the command above. However, more text than that needs to be used to generate syntactically and semantically correct English.

### Generate using an existing model

Supply the name of the model and the serialized tokenizer, and a prompt:

```sh
python3 sLAM/generate.py -n 04-01-2025-05-09-04 -p "This is a test"
This is a test if your favorite software is the news service for the bottom of the 
increasing equipment market is actually plans for their concerns and the narrative 
of the same time i think it was the course of the technology is that the 5th us and 
i think what we are the most youre doing it we do to do that you want what to avoid 
the first amendment and other candidates are not just as the most.
```

## TensorFlow Architecture and Components

The sLAM project implements a small GPT-2-style transformer language model using TensorFlow/Keras. Here's a detailed explanation of the key components and how they work together:

### Model Architecture Overview

The model is a **decoder-only transformer** - a type of neural network architecture that excels at understanding and generating sequential text. Unlike encoder-decoder models used for translation, this architecture focuses purely on text generation by predicting the next token in a sequence.

### Core TensorFlow Components

#### 1. Token and Positional Embeddings

**Token Embeddings**: Convert each word/subword token into a dense vector representation (e.g., 256-dimensional). These embeddings learn to capture semantic meaning - similar words end up with similar vector representations.

**Positional Embeddings**: Since transformers process all tokens simultaneously (unlike RNNs), they need explicit position information. Positional embeddings encode where each token appears in the sequence, allowing the model to understand word order and syntax.

The final input representation is: `Token Embedding + Positional Embedding`

#### 2. Multi-Head Attention Mechanism

The core innovation of transformers is **self-attention**, which allows each token to "attend to" or focus on other relevant tokens in the sequence. Multi-head attention runs several attention operations in parallel:

- **Query (Q)**, **Key (K)**, **Value (V)** matrices: Each token is projected into these three representations
- **Attention computation**: `Attention(Q,K,V) = softmax(QK^T/√d_k)V`
- **Multiple heads**: Different attention heads can focus on different types of relationships (syntax, semantics, etc.)
- **Causal masking**: Ensures the model can only attend to previous tokens during generation, not future ones

#### 3. Transformer Blocks

Each transformer block contains:

- Multi-head attention layer
- Feed-forward network (2 dense layers with GELU activation)
- **Residual connections**: Help with gradient flow during training
- **Layer normalization**: Stabilizes training and improves performance

The model stacks multiple transformer blocks (default: 4 layers) to build increasingly complex representations.

#### 4. Model Parameters Explained

- **vocab_size (50,000)**: Number of unique tokens the model can understand/generate
- **context_size (32)**: Maximum sequence length the model can process at once (the "memory window")  
- **d_model (256)**: Dimensionality of embeddings and internal representations
- **n_heads (4)**: Number of parallel attention heads
- **n_layers (4)**: Number of transformer blocks stacked together
- **d_ff (1024)**: Hidden layer size in the feed-forward networks

### Training Process

#### Data Preparation

1. **Text Cleaning**: Filters high-quality text from datasets (cc_news or wikitext)
2. **Tokenization**: Converts text to integer token IDs using Keras TextVectorization
3. **Sequence Creation**: Sliding window approach creates input/target pairs
   - Input: `[token1, token2, token3, token4]`
   - Target: `[token2, token3, token4, token5]` (next token prediction)

#### Model Training

- **Loss Function**: Sparse Categorical Crossentropy (predicts next token from vocabulary)
- **Optimization**: Adam optimizer with polynomial learning rate decay
- **Mixed Precision**: Uses float16 for faster training while maintaining float32 for stability
- **Regularization**: Dropout (10% by default) prevents overfitting

#### Training Monitoring

The code includes several custom callbacks for monitoring training stability:

- **Numerical Stability Callback**: Detects NaN/infinite values and extreme weights
- **Validation Callback**: Tracks performance on held-out data
- **MLFlow Integration**: Logs metrics, parameters, and model artifacts for experiment tracking

### Text Generation Process

During generation, the model:

1. **Encodes** the input prompt into token IDs
2. **Predicts** probability distribution over all possible next tokens
3. **Applies temperature scaling**: Controls randomness (lower = more deterministic, higher = more creative)
4. **Samples** next token from the probability distribution  
5. **Updates** context window by sliding tokens left and adding the new token
6. **Repeats** until desired length or end token is reached

### Memory and Compute Optimizations

- **GPU Memory Growth**: Prevents TensorFlow from allocating all GPU memory at once
- **Asynchronous Memory Allocation**: Uses CUDA malloc for better GPU memory management  
- **Data Pipeline Optimization**: Uses `tf.data` with prefetching and shuffling for efficient data loading
- **Checkpointing**: Saves model state during training to prevent loss of progress

### Model Scale Comparison

This sLAM model is significantly **smaller** than production models like GPT-3 in several key dimensions:

**Arbitrary Parameter Count:**

- sLAM: ~1-5 million parameters (depending on vocab_size and d_model settings)
- GPT-3: 175 billion parameters (35,000x larger)
- GPT-2 small: 117 million parameters (still 100x larger than sLAM)

**Model Dimensions:**

- sLAM: 256 embedding dimensions, 4 layers, 4 attention heads
- GPT-3: 12,288 embedding dimensions, 96 layers, 96 attention heads
- GPT-2 small: 768 embedding dimensions, 12 layers, 12 attention heads

**Context Window:**

- sLAM: 32 tokens (very limited memory)
- GPT-3: 2,048 tokens (64x larger context)
- Modern models: up to 1M+ tokens

**Training Data:**

- sLAM: Thousands of text samples (megabytes)
- GPT-3: ~45TB of internet text data

**Compute Requirements:**

- sLAM: Trainable on consumer hardware (few GB RAM, optional GPU)
- GPT-3: Required thousands of high-end GPUs and months of training

Despite being much smaller, this architecture demonstrates the core principles behind modern large language models and provides an excellent learning platform for understanding transformer-based text generation without requiring massive computational resources.

### Library and package versions

One of the challenges in writing and running Deep Learning code is how many components there are, and how quickly new versions replace old versions. To get all your component versions aligned start with your computer, which may be a GPU. For example, if it's NVIDIA, what is the recommended version of CUDA? From that version find the recommended version of Tensorflow or PyTorch. Then for that package version what version of Python. An example set of versions, working with an older NVIDIA GPU:

RTX 5000 + CUDA 11.8 + Tensorflow 2.17 + Python 3.8

Then the Python dependencies will follow from the Python version.

*Getting these versions aligned is critical*, because if the versions are out of alignment you may get errors of various kinds that do not reference versions and are difficult to debug, like out-of-memory or data shape errors.

#### Using Tensorflow from a container

Containers may be available that package all the right versions, e.g. CUDA and Python with some framework. In this example we're computing at Texas Advanced Computing Center and downloading a Tensorflow container from NVIDIA:

```sh
srun -N 1 -n 10 -p rtx-dev -t 60:00 --pty bash
module load tacc-apptainer
apptainer pull docker://tensorflow/tensorflow:2.17.0-gpu
```

Or just:

```sh
docker pull tensorflow/tensorflow:2.17.0-gpu
```

Then you can run your script with `singularity`.

```sh
singularity exec --nv tensorflow_2.17.0-gpu.sif python3 scripts/mnist_convnet.py 
```
