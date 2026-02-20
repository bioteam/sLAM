# sLAM

Demonstration code to create a GPT-2-style, generative small LAnguage Model that can be built using personal computing.

This is not for production. You can use this code to learn about Tensorflow, generative language models, preprocessing, training, and model hyperparameters.

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
                    [--min_chunk_len MIN_CHUNK_LEN] -p PROMPT [-v]

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
  -p PROMPT, --prompt PROMPT
                        Prompt
  -v, --verbose         Verbose
```

The code uses *cs_news* (the default) or *wikitext-2-v1* from Hugging Face as training text. *cs_news* is the cleaner of the 2, with less formatting text.

### Default Parameters

- `epochs` (3): Number of complete passes through the entire training dataset
- `vocab_size (50,000)`: Number of unique tokens the model can understand/generate
- `context_size (32)`: Maximum sequence length the model can process at once (the "memory window")  
- `d_model (256)`: Dimensionality of embeddings and internal representations
- `n_heads (4)`: Number of parallel attention heads
- `n_layers (4)`: Number of transformer blocks stacked together
- `d_ff (1024)`: Hidden layer size in the transformer blocks

### Build a model

Download and clean training data from *cs_news*, tokenize it into large chunks, create a model, train the model using context-window-sized slices for 3 epochs, be verbose, and try the given prompt:

```sh
python3 sLAM/make-slam.py --num_datasets 1000 -v --epochs 3 -p "This is a test"
```

The command creates a Keras model using ~1M input tokens and a saved (serialized) tokenizer with the same name, and histograms of chunk lengths and token input numbers. for example:

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

Very little text input is needed to get decent syntax but the semantics are off. As you increase the number of epochs and the number of input tokens the output approaches semantically correct English, for example, after 100 epochs with 10K *cc_news* datasets:

*This is a test of paris ap — french president emmanuel macron is condemning the rally in charlottesville today to neonazis skinheads and ku klux klan members and the white nationalists were met with hundreds of counterprotesters.*

*This is a test of now playing gives you a party to your family and friends who shared with facebook on facebook now playing what would you do with the explosion now playing what might mean for us now playing family.*

## Architecture and Components

Here's a detailed explanation of the key components and how they work together. The model is a type of neural network architecture that can understand and generate sequential text. Unlike encoder-decoder models used for translation, this architecture focuses purely on text generation by predicting the next token in a sequence.

### Token and Positional Embeddings

An embedding is a __numerical vector representation__ or __tensor__ of a discrete item like a token string or token position. The embeddings are initialized with random values within a small range, then __trained__. During training the model adjusts these embeddings by small increments, then tests these new embeddings to see if they can predict the next token better. In this code specifically, each embedding is a 1D tensor (vector) but the sLAM code also uses __3D tensors__. When data is batched for training, each batch contains multiple sequences (in language modeling a __sequence__ refers to an ordered series of tokens):

- Shape: `(batch_size, context_size, d_model)` = `(batch_size, 32, 256)` by default

  - Dimension 1: batch size (e.g., 4 samples processed together)
  - Dimension 2: sequence length (32 tokens)
  - Dimension 3: embedding dimensions (256-dimensional vectors)

There are 2 kinds of embeddings:

*Token Embeddings*: Convert each word token into a dense vector representation (e.g., 256 dimensions). These embeddings are adjusted - "learn" - in training to capture semantic meaning - similar words end up with similar vector representations.

*Positional Embeddings*: Transformers process all tokens simultaneously and need explicit position information. Positional embeddings encode where each token appears in the sequence, enabling the model to predict correct word order and syntax.

Token and positional embeddings are created using __Keras Embedding layers__, which are lookup tables that map indices to vectors.

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
    input_dim=context_size,    # 32 tokens or positions
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

Both embedding layers are the __learnable weights__ that are trained during model training through backpropagation.

### Multi-Head Attention Mechanism

A core innovation of transformers is *self-attention*, which trains each token to "attend to" or relate to other relevant tokens in the sequence.

Attention is a computational mechanism that transforms token embeddings into Query (Q), Key (K), and Value (V) vectors, then computes similarity scores between Q and K to generate weights via softmax, and finally produces a weighted sum of the V vectors. Multi-head attention performs this computation multiple times in parallel with different learned transformations. Position influences attention in two ways: positional embeddings are incorporated into the token representations that become Q, K, and V, and causal masking constrains which positions each token can attend to—preventing attention to future tokens and allowing only self and past token attention.

#### Attention computation

`Attention(Q,K,V) = softmax(QK^T/√d_k)V`

The formula computes a weighted sum of the Values (V). The softmax of (QK^T/√d_k) produces attention weights, a probability distribution showing which keys are most relevant to each query. These weights are then applied to V to produce an output tensor where each position has been enriched with information from all other relevant positions. In the sLAM code, this attention output is fed into a residual connection (added back to the input), then passed through layer normalization, and then into the feed-forward network. So the attention result is the "refined" token representation that incorporates context from other tokens in the sequence.

During the Q-K comparison, some token pairs produce high dot product scores and others produce low scores. Tokens with high scores are given more weight in the final output. What counts as "relevant" (what produces high scores) is learned by the model during training, the Q, K, and V weight matrices are adjusted so that the model learns to assign high scores to tokens that help predict the next token accurately.

Q, K, and V are __computed from the input embeddings__ using learnable __weight matrices__. So the actual weights that get adjusted during backpropagation are those transformation matrices (often called projection matrices). For each input embedding, the model multiplies it by these weight matrices to produce Q, K, and V vectors. During training, backpropagation adjusts these weight matrices so that the resulting Q, K, V vectors encode relationships that help the model predict the next token accurately.

##### Attention and predicting the next word

In a decoder-only model like this the model attends to its own previous tokens to predict the next one. This is called causal self-attention — each token can only see tokens before it, not future ones. If the input is the sequence of tokens *A | cat | sat | on | the* then we want to predict the word after "the". Each token attends to all preceding tokens (including itself), something like this:

1.Each token gets Q, K, V vectors

`"A"   → Q₁, K₁, V₁`
`"cat" → Q₂, K₂, V₂`
`"sat" → Q₃, K₃, V₃`
`"on"  → Q₄, K₄, V₄`
`"the" → Q₅, K₅, V₅   ← this is the query position`

2.Score Q₅ against all previous keys

`score("the" vs "A")   = Q₅ · K₁ = 0.4`
`score("the" vs "cat") = Q₅ · K₂ = 2.1`
`score("the" vs "sat") = Q₅ · K₃ = 0.8`
`score("the" vs "on")  = Q₅ · K₄ = 1.1`
`score("the" vs "the") = Q₅ · K₅ = 3.9  ← attends to itself + context`

Future tokens are masked to -∞ before softmax, so they become 0.

3.Softmax → attention weights

`weights ≈ [0.02, 0.12, 0.05, 0.08, 0.73]`

The model leans heavily on the current "the" (it's a determiner, so a noun likely follows) but also picks up signal from "cat" (a noun came after the first "the").

4.Weighted sum → context vector → predict next token

`context = 0.02·V₁ + 0.12·V₂ + 0.05·V₃ + 0.08·V₄ + 0.73·V₅`

This context vector feeds into a linear layer + softmax over the vocabulary. The model outputs a probability distribution — ideally high probability on tokens like "mat", "floor", "rug", etc.

What training does
The correct next token ("mat") is known. The cross-entropy loss is computed, and backpropagation adjusts Wq, Wk, Wv so that over many examples:

- "A" preceding a noun learns to attend to prior nouns for context
- "sat on the __" learns to up-weight location/surface nouns

### Model architecture

Input tokens are converted to token embeddings vectors (initialized randomly and adjusted during training), combined with positional embeddings, and passed through a dropout layer for data regularization. Then 4 transformer blocks are stacked, each containing: (1) a multi-head attention layer with 4 heads and causal masking that computes relevance-weighted combinations of tokens, (2) a residual connection adding the attention output back to its input, (3) layer normalization, (4) a feed-forward network (FFN) with two dense layers (first expands to `d_ff` - 1024 dimensions by default - with GELU activation, second contracts back to `d_model` - 256 dimensions by default - linearly), (5) dropout, and (6) another residual connection plus layer normalization. Finally, a dense layer projects the output to the vocabulary size (default: 50,000) to produce logits for predicting the next token. Data flows through as 3D tensors of shape (batch_size, sequence_length, embedding_dimension), and all weights are learned through backpropagation during training.

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

The model has 4 blocks, by default, specified by the `n_layers` parameter. Each transformer block contains the following components, in order:

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

#### Data Preparation or preprocessing

An unappreciated detail to a novice is that all input to a neural network, training or inference, is some form of matrix filled with numbers, integer or float.

1. *Text Cleaning*: Filters high-quality text from datasets (*cc_news* or *wikitext*)
2. *Tokenization*: Converts text to integer token IDs using Keras TextVectorization
3. *Sequence Creation*: Sliding window approach creates input/target pairs for next token prediction, for example:
   - Input: `[token1, token2, token3, token4]`
   - Target: `[token2, token3, token4, token5]`

#### Model Training

Within a single epoch:

1. __Batching__: Training data is divided into batches (default batch_size=4 in `slam.py`)
2. __Forward Pass__: For each batch, the model makes predictions
3. __Loss Calculation__: Compare predictions to target tokens using `SparseCategoricalCrossentropy`
4. __Backward Pass (Backpropagation)__: Compute gradients
5. __Parameter Update__: Adam optimizer updates all weights
6. __Repeat__: Steps 2-5 repeat for each batch until all training data is processed
7. __Validation__: After all batches, model is evaluated on validation data
8. __Epoch Complete__: One full pass is done

For example in `slam.py` if you have 10,000 training samples and batch_size=4:

- Steps per epoch = 10,000 ÷ 4 = 2,500 steps
- With epochs=3, training runs 3 complete passes = 7,500 steps

##### Embeddings

The embeddings are the actual learnable weights that get adjusted during training. An embedding layer is created initially:

```python
layers.Embedding(input_dim=50000, output_dim=256)
```

This creates a weight matrix of shape (50,000, 256), i.e. 50,000 rows (one per token) × 256 columns (the embedding dimension) = 12,800,000 individual float values. These 12.8M floats are the learnable parameters.

##### During training

- These 256 floats for each token start as random values
- Backpropagation adjusts each individual float based on the loss gradient
- Over time, tokens that appear in similar contexts end up with similar embedding vectors (similar patterns of 256 floats)

For example:

- Token "cat" might initially be: [0.02, -0.15, 0.08, ..., 0.12]
- Token "dog" might initially be: [-0.11, 0.09, -0.03, ..., 0.18]

After training, if they appear in similar contexts, their 256 floats adjust to become more similar.

### The loss function

The loss function measures __how wrong the model's predictions are__ during training and serves as the signal that guides the learning process. In training flow:

- Forward pass: model outputs logits for next token prediction
- Loss calculation: `SparseCategoricalCrossentropy` compares logits to actual next token ID
- Backward pass: gradients flow back through the network
- Weight update: optimizer adjusts weights to reduce future loss
- Early stopping: monitors val_loss and stops training when it stops improving
- Checkpoints: saves models when val_loss is lowest

In essence, the loss function is the __feedback mechanism__ that tells the model how to learn.

### Optimization

Adam (Adaptive Moment Estimation) is an optimization algorithm used to update the model's weights during training based on the gradients computed from the loss function.

```python
self.optimizer = tf.keras.optimizers.Adam(
    learning_rate=lr_schedule, epsilon=1e-8
)
```

The Adam optimizer stores and uses data from previous steps for optimization using exponentially decaying moving averages, not by storing all raw results from every past epoch or step. This memory is maintained across epochs as part of the optimizer's internal state. Here is how Adam uses past data:

- Momentum (First Moment Estimate): Adam maintains a running exponential moving average of the past gradients, often referred to as the "first moment" vector (denoted as <i>m</i>). This helps to smooth out the optimization path and maintain velocity in consistent directions, leading to faster convergence and reduced oscillation.
- Adaptive Learning Rates (Second Moment Estimate): Adam also tracks an exponential moving average of the squared gradients, known as the "second moment" vector (denoted as <i>v</i>). This information is used to adapt the learning rate for each individual parameter of the model, allowing for larger steps for infrequent parameters and smaller steps for frequent ones.
- Bias Correction: The moving averages <i>m</i> and <i>v</i> are initialized with zeros and are therefore biased towards zero, especially during the initial iterations of training. Adam applies a bias correction mechanism to these estimates to ensure they are more accurate, particularly in the early stages of training.

<i>m</i> and <i>v</i> are updated at every training step and persist throughout the entire training process, allowing the optimizer to intelligently navigate the complex loss landscape. The memory requirement for this is relatively low as it only involves storing two moving average vectors that are the same size as the model's parameters.

#### Gradients

The gradients inform the training code on how to adjust the internal learnable weights, or parameters. Gradients in a neural network are vectors of partial derivatives that measure how much the network's loss (error) changes with respect to its weights and biases. They represent the slope of the cost function, pointing in the direction of the steepest ascent. By computing gradients via backpropagation, optimizers update parameters in the opposite direction (gradient descent) to minimize error and improve model performance.

The gradients indicate the direction and rate at which parameters should be adjusted to reduce error. They point "uphill" towards higher loss, which is why algorithms move in the negative gradient direction to go "downhill" (minimize loss). A large gradient indicates a steep slope (requiring significant updates), while a small gradient indicates a flat region. The backpropagation algorithm calculates the gradients for every parameter by traversing the network backward from the output layer to the input layer.

How it fits into training:

1. Forward pass: Model predicts next token
2. Loss calculated via `SparseCategoricalCrossentropy`
3. __Adam computes gradients__ using backpropagation
4. __Adam updates weights__ using adaptive per-parameter learning rates
5. Process repeats for each batch
6. Early stopping monitors validation loss and stops when no improvement

Adam is the default choice for training modern neural networks including language models like sLAM because it combines the benefits of momentum-based methods with adaptive learning rates.

#### FFN weights

The FFN in sLAM is two dense (fully connected) layers:

```python
layers.Dense(self.d_ff, activation="gelu")  # expand: 256 → 1024
layers.Dense(self.d_model)                  # contract: 1024 → 256
```

This expands the representation to a larger dimension (1024), applies a non-linear activation (GELU), then contracts back down to the model dimension (256). This expansion-contraction gives the model extra capacity to transform the representation in ways attention alone can't.

A rough intuition for the division of labor:

- Attention — mixes information across tokens (which tokens relate to which)
- FFN — transforms the representation of each token independently (no cross-token interaction)

The FFN weights are also learned during training via backpropagation, just like the embedding and attention weight matrices. Both FFN weights and embeddings are weight matrices updated by backpropagation — but they work differently.

Each embedding corresponds to a specific token, but the FFN weights are applied via matrix multiplication to every token's vector, they are not tied to specific tokens — the same weights are applied to every position
A concrete way to see the difference:

`Embedding:  token_id=42  →  look up row 42  →  [0.3, -0.1, 0.8, ...]`
`FFN:        vector       →  multiply by W   →  new transformed vector`

The deeper similarity is that both are just matrices of floats that get adjusted by Adam during training. In that sense all learned parameters in a neural network are matrices of numbers updated by gradient descent. The difference is in how they're used during the forward pass.

### The Forward and Backward Pass

A simplified view:

- Forward:  embeddings → attention → ... → loss
- Backward: loss → gradients → optimizer updates {Wq, Wk, Wv, FFN weights, embedding table}

#### Backward pass for training/learning

- The loss is computed (cross-entropy vs. the correct next token)
- Gradients flow backward through the attention mechanism all the way back to the embeddings
- The Adam optimizer then updates everything — both the embedding table values AND the Q, K, V weight matrices

Attention doesn't modify the embedding table. Attention is the messenger that carries gradient signal back to the embeddings during backpropagation — it's the path through which the embeddings learn what they should represent. But the actual update to the embedding floats is done by the optimizer, not by attention itself.

A concrete way to think about it: if "cat" and "mat" always appear in similar attention patterns, backpropagation will adjust their embedding vectors to be more similar — but attention didn't do that directly, it just created the context in which the loss signal could flow back and cause those updates.

#### Forward pass for prediction

- The embedding lookup produces vectors
- Attention transforms those vectors into contextual representations (a weighted sum of Values)
- The embeddings themselves are unchanged — attention just reads from them

#### Training Monitoring

Checkpoints serve as __intermediate saves of the model's learned weights__ during training, enabling recovery, model selection, and efficient disk management. A __Callback__ is a mechanism that hooks into the training process at specific events (epoch end, batch end, etc.). In `slam.py`, callbacks are custom classes like `ValidationPrintCallback` and `SmartCheckpointCallback` that inherit from `tf.keras.callbacks.Callback`. They execute custom logic during different stages of training.
For example, The `SmartCheckpointCallback` manages the creation and lifecycle of checkpoint files, saving them at intervals and keeping only the N best checkpoints based on validation loss.

The custom callbacks for monitoring training stability:

- `ValidationPrintCallback`: Tracks performance on held-out data
- `SmartCheckpointCallback`: Saves model state during training
- `EarlyStopping`: when training stops due to no improvement, the model weights are restored to the best checkpoint

### What are the "weights"?

A useful intuition: attention decides where to look (dynamic), while the embeddings determine what that means (static, learned at training time). Both are essential — attention alone with random weights predicts nothing useful, and embeddings without attention can't model sequential dependencies.

Embeddings are static lookup tables learned during training. They provide the initial representation of each token going into the transformer blocks. The token embedding matrix (50,000 × 256 = 12.8M parameters) encodes what each token "means" in isolation and the positional embeddings add sequence order information.

The attention content (Q, K, V weight matrices) are also learned weights, but they compute dynamic, context-dependent transformations. The Q, K, V projection matrices transform the embeddings into vectors that can be compared against each other. The attention scores (softmax of QKᵀ/√d_k) determine which tokens' Values get weighted together. The final prediction comes from the dense output layer projecting the last hidden state → vocabulary logits. So the full chain is:

  → Token Embedding (learned weights)
  → Positional embeddings (learned weights)
  → Attention (Q,K,V learned weight matrices)
  → FFN (learned weight matrices)
  → Output dense layer (learned weights)
  → Logits over 50,000 tokens

### Text Generation Process

During generation, the model:

1. *Encodes* the input prompt into token IDs
2. *Predicts* probability distribution over all possible next tokens
3. *Applies temperature scaling*: Controls randomness (lower = more deterministic, higher = more creative)
4. *Samples* next token from the probability distribution
5. *Updates* context window by sliding tokens left and adding the new token
6. *Repeats* until desired length or end token is reached

### Model Size Comparisons

This sLAM models I've made are much smaller than production models like GPT* in most respects:

*Arbitrary Parameter Count:*

- sLAM: ~1-5M parameters (depending on *vocab_size* and *d_model* settings)
- GPT-2 small: 117M parameters
- GPT-3: 175B parameters

*Model Dimensions:*

- sLAM: 256 embedding dimensions, 4 transformer blocks, 4 attention heads
- GPT-2 small: 768 embedding dimensions, 12 transformer blocks, 12 attention heads
- GPT-3: 12,288 embedding dimensions, 96 transformer blocks, 96 attention heads

*Context Window:*

- sLAM: 32 tokens
- GPT-2: 1024 tokens
- GPT-3: 2,048 tokens

*Training Data:*

- sLAM: Thousands of text samples (megabytes)
- GPT-2: 40GB of text data
- GPT-3: 570GB of text data

*Compute Requirements:*

- sLAM: Trainable on consumer hardware (few GB RAM, optional GPU)
- GPT-2: 8 TPUs v2 (comparable to 32-64 NVIDIA V100s)
- GPT-3: Required thousands of high-end GPUs and months of training

### Library and package versions

One of the challenges in writing and running TensorFlow code is how many dependencies there are, and how quickly new versions replace old versions. To get all your versions aligned start with your computer, which may be a GPU. For example, if it's NVIDIA, what is the recommended version of CUDA? From that version find the recommended version of Tensorflow or PyTorch. Then for that package version what version of Python. An example set of versions, working with an older NVIDIA GPU:

RTX 5000 + CUDA 11.8 + Tensorflow 2.17 + Python 3.8

Then all the other Python dependencies (e.g. pandas, numpy) will follow from the Python version.

*Getting these versions aligned is critical*, because if the versions are out of alignment you create executable code but get errors of various kinds that do not reference versions and are difficult to debug, like out-of-memory or data shape errors.

#### Using Tensorflow from a container

Containers may be available that package all the right versions, e.g. of CUDA, Python, and Tensorflow. In this example we're computing at the Texas Advanced Computing Center and downloading a Tensorflow container from NVIDIA:

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
