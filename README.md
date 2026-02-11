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
usage: make-slam.py [-h] [-t TEXT_PERCENTAGE] [--context_size CONTEXT_SIZE] [-n NAME] [--temperature TEMPERATURE] [--epochs EPOCHS] [--d_model D_MODEL] [-d {wikitext-2-v1,cc_news}] [--num_datasets NUM_DATASETS]
                    [--min_chunk_len MIN_CHUNK_LEN] [--use_mlflow] -p PROMPT [-v]

options:
  -h, --help            show this help message and exit
  -t TEXT_PERCENTAGE, --text_percentage TEXT_PERCENTAGE
                        Percentage of wikitext-2-v1 used to make dataset
  --context_size CONTEXT_SIZE
                        Context size
  -n NAME, --name NAME  Name used to save files, default is timestamp of completion
  --temperature TEMPERATURE
                        Temperature used for generation
  --epochs EPOCHS       Number of epochs
  --d_model D_MODEL     Number of embedding dimensions
  -d, --download {wikitext-2-v1,cc_news}
                        Dataset to download
  --num_datasets NUM_DATASETS
                        Number of datasets to download from cc_news
  --min_chunk_len MIN_CHUNK_LEN
                        Minimum length of cc_news chunk to use for training
  --use_mlflow          Use MLFlow
  -p PROMPT, --prompt PROMPT
                        Prompt
  -v, --verbose         Verbose
```

The code uses *cs_news* (the default) or *wikitext-2-v1* from Hugging Face as training text. *cs_news* is the cleaner of the 2, with less formatting text.

### Default Parameters

- *vocab_size (50,000)*: Number of unique tokens the model can understand/generate
- *context_size (32)*: Maximum sequence length the model can process at once (the "memory window")  
- *d_model (256)*: Dimensionality of embeddings and internal representations
- *n_heads (4)*: Number of parallel attention heads
- *n_layers (4)*: Number of transformer blocks stacked together
- *d_ff (1024)*: Hidden layer size in the feed-forward networks

### Build a model

Download and clean training data from *cs_news*, tokenize it into large chunks, create a model, train the model using context-window-sized slices for 3 epochs, be verbose, and try the given prompt:

```sh
python3 sLAM/make-slam.py --num_datasets 1000 -v --epochs 3 -p "This is a test"
```

The command creates a Keras model (~1M input tokens) and a saved (serialized) tokenizer with the same name, and histograms of chunk lengths and token input numbers. for example:

```sh
-rw-r--r--   332M Apr  1 05:09 04-01-2025-05-09-04.keras
-rw-r--r--    58K Apr  1 05:09 04-01-2025-05-09-04.pkl
-rw-r--r--    19K Mar 31 16:04 chunk_length_distribution.png
-rw-r--r--    19K Mar 31 16:05 token_number_distribution.png
```

One epoch takes about ~1 hour on a Mac M1 laptop (32 GB RAM) with the command above. However, more text than that needs to be used to generate syntactically and semantically correct English.

### Generate using an existing model

Supply the name of the model and the serialized tokenizer, and a prompt:

```sh
python3 sLAM/generate.py -n 04-01-2025-05-09-04 -p "This is a test"
This is a test of now playing this case means he may be caught trying to arrest him by name and his wife and their father i really wanted to play for him and to get even better.
```

### Playing with epochs and input

Very little input is needed to get decent syntax but the semantics are off. As you increase the number of epochs and the number of input tokens the output approaches semantically correct English, for example, after 100 epochs with 10K *cc_news* datasets:

*This is a test of paris ap — french president emmanuel macron is condemning the rally in charlottesville today to neonazis skinheads and ku klux klan members and the white nationalists were met with hundreds of counterprotesters.*

## Architecture and Components

Here's a detailed explanation of the key components and how they work together. The model is a *decoder-only transformer* - a type of neural network architecture that can understand and generate sequential text. Unlike encoder-decoder models used for translation, this architecture focuses purely on text generation by predicting the next token in a sequence.

### Token and Positional Embeddings

An embedding is a __numerical vector representation__ of a discrete item (like a token or position). The embeddings are initialized with random values within a small range, then __trained__. During training via backpropagation, the model adjusts these embedding vectors to represent tokens and positions in ways that help predict the next token accurately. The information comes from learning patterns in the training data.
An embedding is a __tensor__ (a numerical array), and in this code specifically, each embedding is a 1D tensor (vector).

The sLAM code uses __3D tensors__. When data is batched for training, each batch contains multiple sequences (in language modeling a __sequence__ refers to an ordered series of tokens):

- Shape: `(batch_size, context_size, d_model)` = `(batch_size, 32, 256)` by default

  - Dimension 1: batch size (e.g., 4 samples processed together)
  - Dimension 2: sequence length (32 tokens)
  - Dimension 3: embedding dimensions (256-dimensional vectors)

These 3D tensors flow through the entire model.

*Token Embeddings*: Convert each word token into a dense vector representation (e.g., 256-dimensions). These embeddings learn to capture semantic meaning - similar words end up with similar vector representations.

*Positional Embeddings*: Transformers process all tokens simultaneously and need explicit position information. Positional embeddings encode where each token appears in the sequence, allowing the model to understand word order and syntax.

Token and positional embeddings are created using __Keras Embedding layers__, which are lookup tables that map indices to learned vectors.

In the `TokenAndPositionEmbedding` class in the code:

__Token Embeddings:__

This creates a lookup table with 50,000 rows (one for each token in the vocabulary), where each row is a 256-dimensional vector. When you look up a token ID, you get its corresponding embedding vector.

```python
self.token_emb = layers.Embedding(
    input_dim=vocab_size,      # 50,000 tokens
    output_dim=d_model,        # 256 dimensions
    name="token_embeddings"
)
```

__Positional Embeddings:__

This creates another lookup table with 32 rows (one for each position in the sequence), where each row is also a 256-dimensional vector. When you look up a position (0-31), you get its corresponding embedding.

```python
self.pos_emb = layers.Embedding(
    input_dim=context_size,    # 32 positions
    output_dim=d_model,        # 256 dimensions
    name="position_embeddings"
)
```

__Combined in the call method:__

```python
token_embeddings = self.token_emb(inputs)        # Get token vectors
position_embeddings = self.pos_emb(positions)    # Get position vectors
return token_embeddings + position_embeddings    # Add them together
```

Both embedding layers have __learnable weights__ that are trained during model training through backpropagation.

### Multi-Head Attention Mechanism

The core innovation of transformers is *self-attention*, which trains each token to "attend to" or relate to other relevant tokens in the sequence.

Attention is a computational mechanism that transforms token embeddings into Query (Q), Key (K), and Value (V) vectors, then computes similarity scores between Q and K to generate weights via softmax, and finally produces a weighted sum of the V vectors. Multi-head attention performs this computation multiple times in parallel with different learned transformations. Position influences attention in two ways: positional embeddings are incorporated into the token representations that become Q, K, and V, and causal masking constrains which positions each token can attend to—preventing attention to future tokens and allowing only self and past token attention.

#### Attention computation

`Attention(Q,K,V) = softmax(QK^T/√d_k)V`

The formula computes a weighted sum of the Values (V). The softmax of (QK^T/√d_k) produces attention weights, a probability distribution showing which keys are most relevant to each query. These weights are then applied to V to produce an output tensor where each position has been enriched with information from all other relevant positions. In the sLAM code, this attention output is fed into a residual connection (added back to the input), then passed through layer normalization, and then into the feed-forward network. So the attention result is the "refined" token representation that incorporates context from other tokens in the sequence.

During the Q-K comparison, some token pairs produce high dot product scores and others produce low scores. Tokens with high scores are given more weight in the final output. What counts as "relevant" (what produces high scores) is learned by the model during training, the Q, K, and V weight matrices are adjusted so that the model learns to assign high scores to tokens that help predict the next token accurately.

Q, K, and V are __computed from the input embeddings__ using learnable __weight matrices__. So the actual weights that get adjusted during backpropagation are those transformation matrices (often called projection matrices). For each input embedding, the model multiplies it by these weight matrices to produce Q, K, and V vectors. During training, backpropagation adjusts these weight matrices so that the resulting Q, K, V vectors encode relationships that help the model predict the next token accurately.

### Model architecture

The sLAM model architecture is a decoder-only transformer. Input token IDs are converted to token embeddings (256-dimensional vectors initialized randomly and learned during training), combined with positional embeddings of the same dimension, and passed through a dropout layer for data regularization. Then 4 transformer blocks are stacked, each containing: (1) a multi-head attention layer with 4 heads and causal masking that computes relevance-weighted combinations of tokens, (2) a residual connection adding the attention output back to its input, (3) layer normalization, (4) a feed-forward network with two dense layers (first expands to 1024 dimensions with GELU activation, second contracts back to 256 dimensions linearly), (5) dropout, and (6) another residual connection plus layer normalization. Finally, a dense layer projects the output to the vocabulary size (50,000) to produce logits for predicting the next token. Data flows through as 3D tensors of shape (batch_size, sequence_length, embedding_dimension), and all weights are learned through backpropagation during training.

So the complete flow is:

```javascript
Input → Token+Position Embeddings → Dropout → [Transformer Block 1] → [Transformer Block 2] → [Transformer Block 3] → [Transformer Block 4] → Output Dense Layer → Logits
```

The embedding and dropout layers prepare the data, the blocks do the main processing, and the final dense layer produces the prediction scores for each token in the vocabulary.

#### Before and after the transformers

There are layers both __before__ and __after__ the transformer blocks:

__Before the transformer blocks:__

1. __Input layer__ - accepts the token IDs
2. __TokenAndPositionEmbedding layer__ - combines token embeddings with positional embeddings
3. __Dropout layer__ - randomly drops 10% of values for regularization

__After the transformer blocks:__

__Dense layer__ - projects the output to the vocabulary size to produce logits for predicting the next token

#### Dropout layer

Dropout is a __regularization technique__ that prevents overfitting. During training, it randomly sets a percentage of values (10% by default in sLAM) to zero. This forces the model to learn more robust features by not relying on any single activation value. It's like training with incomplete information—the model learns to work with different random subsets of neurons, which makes it generalize better to new data. During generation/inference, dropout is not applied.

Dropout is used in three places: before the blocks, within each block's attention mechanism, and after each block's feed-forward network.

#### Transformer Blocks

The model has 4 blocks, by default, specified by the *n_layers* parameter. Each transformer block contains the following components, in order:

- __Multi-head attention layer__ with causal masking
- __Residual connection__ adding attention output back to input
- __Layer normalization__
- __Feed-forward network__ with two dense layers:
  - First dense layer with GELU activation: `layers.Dense(self.d_ff, activation="gelu")(x)`
  - Second dense layer with no activation (linear): `layers.Dense(self.d_model)(ff_output)`
- __Dropout layer__ for regularization (10% by default, applied after feed-forward)
- __Residual connection__ adding feed-forward output back to input
- __Layer normalization__

So the flow in each transformer block is:

```javascript
Input → Attention → Add (residual) → LayerNorm → Feed-forward → Add (residual) → LayerNorm → Output
```

##### Layer Normalization

Layer normalization normalizes the activations (output values) of a layer to have a mean of 0 and standard deviation of 1. This keeps the values from becoming too large or too small, which:

- Prevents training from becoming unstable
- Allows for higher learning rates

Layer normalization occurs in two places within each transformer block:

After the attention + residual connection:

```python
x = layers.Add()([x, attn_output])
x = layers.LayerNormalization(epsilon=1e-6)(x)  # <-- HERE
```

This normalizes the output before sending it to the feed-forward network.

After the feed-forward + residual connection:

```python
x = layers.Add()([x, ff_output])
x = layers.LayerNormalization(epsilon=1e-6)(x)  # <-- AND HERE
```

This normalizes the output at the end of the block before it goes to the next block (or output layer).

It's different from batch normalization (which normalizes across the batch dimension)—layer normalization normalizes across the feature dimension for each individual sample, which is more suitable for transformers and sequence models.

So the feed-forward network expands to `d_ff` dimensions (1024 by default) with GELU, then contracts back down to `d_model` dimensions (256 by default) without an activation function.

In the code, residual connections are implemented using `layers.Add()`, which literally adds the original input to the processed output:

1. After the multi-head attention layer: `x = layers.Add()([x, attn_output])`

   - This adds the attention output to the original input

2. After the feed-forward network: `x = layers.Add()([x, ff_output])`

   - This adds the feed-forward output back to the input it received

These additions create "shortcuts" or "skip connections" that allow gradients to flow more effectively during backpropagation, which improves training stability and allows deeper networks to learn better.

### Training Process

#### Data PreparationQQ

1. *Text Cleaning*: Filters high-quality text from datasets (cc_news or wikitext)
2. *Tokenization*: Converts text to integer token IDs using Keras TextVectorization
3. *Sequence Creation*: Sliding window approach creates input/target pairs for next token prediction, for example:
   - Input: `[token1, token2, token3, token4]`
   - Target: `[token2, token3, token4, token5]`

#### Model Training

- *Loss Function*: Sparse Categorical Crossentropy (predicts next token from vocabulary)
- *Optimization*: Adam optimizer with polynomial learning rate decay
- *Mixed Precision*: Uses float16 for faster training while maintaining float32 for stability
- *Regularization*: Dropout (10% by default) prevents overfitting

#### Training Monitoring

The code includes several custom callbacks for monitoring training stability:

- *Numerical Stability Callback*: Detects NaN/infinite values and extreme weights
- *Validation Callback*: Tracks performance on held-out data
- *MLFlow Integration*: Logs metrics, parameters, and model artifacts for experiment tracking
- *Checkpointing*: Saves model state during training to prevent loss of progress

### Text Generation Process

During generation, the model:

1. *Encodes* the input prompt into token IDs
2. *Predicts* probability distribution over all possible next tokens
3. *Applies temperature scaling*: Controls randomness (lower = more deterministic, higher = more creative)
4. *Samples* next token from the probability distribution
5. *Updates* context window by sliding tokens left and adding the new token
6. *Repeats* until desired length or end token is reached

### Model Scale Comparison

This sLAM models I've made are significantly smaller than production models like GPT-3 in most respects:

*Arbitrary Parameter Count:*

- sLAM: ~1-5 million parameters (depending on *vocab_size* and *d_model* settings)
- GPT-2 small: 117 million parameters
- GPT-3: 175 billion parameters

*Model Dimensions:*

- sLAM: 256 embedding dimensions, 4 layers, 4 attention heads
- GPT-3: 12,288 embedding dimensions, 96 layers, 96 attention heads
- GPT-2 small: 768 embedding dimensions, 12 layers, 12 attention heads

*Context Window:*

- sLAM: 32 tokens (very limited memory)
- GPT-3: 2,048 tokens (64x larger context)
- Modern models: up to 1M+ tokens

*Training Data:*

- sLAM: Thousands of text samples (megabytes)
- GPT-3: ~45TB of internet text data

*Compute Requirements:*

- sLAM: Trainable on consumer hardware (few GB RAM, optional GPU)
- GPT-3: Required thousands of high-end GPUs and months of training

### Library and package versions

One of the challenges in writing and running Deep Learning code is how many components there are, and how quickly new versions replace old versions. To get all your component versions aligned start with your computer, which may be a GPU. For example, if it's NVIDIA, what is the recommended version of CUDA? From that version find the recommended version of Tensorflow or PyTorch. Then for that package version what version of Python. An example set of versions, working with an older NVIDIA GPU:

RTX 5000 + CUDA 11.8 + Tensorflow 2.17 + Python 3.8

Then all the other Python dependencies (e.g. pandas, numpy) will follow from the Python version.

*Getting these versions aligned is critical*, because if the versions are out of alignment you may get errors of various kinds that do not reference versions and are difficult to debug, like out-of-memory or data shape errors.

#### Using Tensorflow from a container

Containers may be available that package all the right versions, e.g. of CUDA, Python, and Tensorflow. In this example we're computing at Texas Advanced Computing Center and downloading a Tensorflow container from NVIDIA:

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
