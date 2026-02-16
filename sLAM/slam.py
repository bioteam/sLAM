import os

# Enable asynchronous memory allocation
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
import tensorflow as tf

import numpy as np
import glob
import random
import sys
import time
import pickle
import re
import os
import nltk
from tensorflow.keras import layers  # type: ignore
from tensorflow.keras import Model  # type: ignore
from tensorflow.keras.optimizers.schedules import PolynomialDecay  # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint  # type: ignore  # noqa: F401
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore  # noqa: F401
from tensorflow.keras.losses import SparseCategoricalCrossentropy  # type: ignore
from sklearn.model_selection import train_test_split

tf.keras.mixed_precision.set_global_policy("mixed_float16")


@tf.keras.utils.register_keras_serializable(package="sLAM")
class TokenAndPositionEmbedding(layers.Layer):
    """
    TokenAndPositionEmbedding

    Combines token embeddings with learned positional embeddings.

    Arguments:
        vocab_size -- Size of the vocabulary/token dictionary
        context_size -- Maximum sequence length for input contexts
        d_model -- Dimensionality of the embedding space

    Returns:
        Embedded token representations with positional information
    """

    def __init__(self, vocab_size, context_size, d_model, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.d_model = d_model
        self.token_emb = layers.Embedding(
            input_dim=vocab_size, output_dim=d_model, name="token_embeddings"
        )
        self.pos_emb = layers.Embedding(
            input_dim=context_size,
            output_dim=d_model,
            name="position_embeddings",
        )

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        position_embeddings = self.pos_emb(positions)
        token_embeddings = self.token_emb(inputs)
        return token_embeddings + position_embeddings

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "context_size": self.context_size,
                "d_model": self.d_model,
            }
        )
        return config


class slam_builder:
    """
    A simple language model using transformer decoder-only architecture.
    Hyperparameters are set here before training (parameters are learned in the training,
    e.g. weights, values in the embedding matrices, coefficients in linear regression).
    """

    def __init__(
        self,
        verbose: bool = False,
        name: str = None,  # type: ignore
        vocab_size: int = 50000,
        context_size: int = 0,
        # Same as embedding_dim:
        d_model: int = 0,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int = 1024,
        dropout_rate: float = 0.1,
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 5e-5,
        temperature: float = 0,
        stride: int = 4,
        download: str = None,  # type: ignore
        num_rows: int = 0,
        optimizer: str = "adam",
    ):
        """__init__

        Keyword Arguments:
            verbose -- Print out details (default: False)
            name -- Name identifier for the model (default: {None})
            vocab_size -- Size of the vocabulary/token dictionary (default: {50000})
            context_size -- Maximum sequence length for input contexts (default: 32)
            d_model -- Dimensionality of the model's embeddings (default: 256)
            n_layers -- Number of transformer layers in the model (default: 4)
            n_heads -- Number of attention heads in each transformer layer (default: 4)
            d_ff -- Dimensionality of the feed-forward network (default: 1024)
            dropout_rate -- Rate of dropout for regularization (default: 0.1)
            epochs -- Number of training epochs (default: 3)
            batch_size -- Number of samples per training batch (default: 4)
            learning_rate --
            temperature -- degree of randomness in generation (default: 0.7)
            optimizer -- optimizer to use (default: "adam")
        """
        self.verbose = verbose
        self.name = name
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.stride = stride
        self.num_rows = num_rows
        self.download = download
        self.optimizer = optimizer

        if not self.name:
            self.name = time.strftime("%m-%d-%Y-%H-%M-%S", time.localtime())

        self.token_ids = list()

        """ Set memory growth to avoid OOM issues """
        gpus = tf.config.list_physical_devices("GPU")
        if len(gpus) > 0:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

    def transformer_block(self, x):
        """transformer_block

        A standard transformer block with multi-head attention and feed-forward network.

        Arguments:
            x -- Input tensor to the transformer block

        Returns:
            x -- Transformed tensor with the same shape as the input but processed through
                 self-attention and feed-forward layers

        Components of each transformer block:
        - Multi-head attention layer with causal masking
        - Residual connection adding attention output back to input
        - Layer normalization
        - Feed-forward network (2 dense layers with GELU activation on first layer)
        - Dropout layer for regularization (applied after feed-forward)
        - Second residual connection adding feed-forward output back to input
        - Layer normalization


        Purpose of the transformer_block method:

        1. Layer Initialization: When transformer blocks are created, the layers (attention, feed-forward
        networks, normalization layers) are initialized with random weights following some initialization
        strategy (likely Xavier/Glorot or He initialization).
        2. Parameter Setup: The transformer_block method would set up the multi-head attention mechanisms,
        feed-forward networks, and layer normalization components.
        3. Architecture Construction: The layers are stacked together to form the complete transformer
        architecture.

        Multi-Head Attention

        Multi-head attention is a key component of transformer models that allows them to focus on different parts of input sequences simultaneously.
        Rather than having a single attention mechanism (one "head"), multi-head attention runs multiple attention operations in
        parallel. Each head can focus on different aspects of the input. The process:

        1. Input Transformation: The input is projected into multiple sets of queries (Q), keys (K), and values (V) using different learned projection matrices

        2. Parallel Attention: Each head performs its own scaled dot-product attention:
            a. Computes compatibility between queries and keys
            b. Applies softmax to get attention weights
            c. Takes weighted sum of values

        3. Concatenation: The outputs from all heads are concatenated together

        4. Final Projection: The concatenated output goes through a final linear projection


        Attention is a mechanism that allows neural networks to focus selectively on relevant parts of input data when performing a task.
        It mimics human cognitive attention by dynamically weighting the importance of different elements in a sequence or set of features.

        How Attention Works:

        Query, Key, Value (QKV) Framework:
        Query: What the model is looking for
        Key: What could be matched against
        Value: Information to be extracted if there's a match

        Attention Computation:

        Calculate similarity/relevance scores between query and each key
        Apply softmax to convert scores to probabilities (weights)
        Produce weighted sum of values based on these weights
        Mathematical Representation

        Attention(Q, K, V) = softmax(QK^T/√d_k)V

        Where:

        Q = query matrix
        K = key matrix
        V = value matrix
        d_k = dimension of keys (scaling factor)

        Types of Attention

        Self-Attention:
        Each position attends to all positions in the same sequence
        Allows modeling relationships between elements within the same input
        Core component in Transformers and foundation for modern LLMs

        Cross-Attention:
        Attends between elements of different sequences (e.g., source and target in translation)
        Common in encoder-decoder architectures

        Multi-Head Attention:
        Runs multiple attention mechanisms in parallel
        Each "head" can focus on different aspects of relationships
        Outputs are concatenated and linearly transformed

        Why Attention Matters

        - Handles variable-length inputs without information loss
        - Captures long-range dependencies that RNNs struggle with
        - Provides interpretability through attention weights visualization
        - Enables parallelization unlike sequential RNN processing
        - Solves vanishing gradient problems in long sequences

        Historical Impact

        Revolutionized NLP after introduction in "Attention Is All You Need" paper
        Enabled development of Transformers, BERT, GPT and other modern architectures
        Largely replaced RNNs and LSTMs for sequence modeling tasks

        Attention has become one of the most fundamental building blocks in modern DL architectures,
        particularly for any task involving sequential or structured data.
        """
        # Multi-head attention
        attn_output = layers.MultiHeadAttention(
            num_heads=self.n_heads,
            key_dim=self.d_model // self.n_heads,
            dropout=self.dropout_rate,
        )(x, x, x, use_causal_mask=True)

        # Residual connection and layer norm
        x = layers.Add()([x, attn_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)

        # Feed forward network
        ff_output = layers.Dense(self.d_ff, activation="gelu")(x)
        ff_output = layers.Dense(self.d_model)(ff_output)
        ff_output = layers.Dropout(self.dropout_rate)(ff_output)

        # Second residual connection and layer norm
        x = layers.Add()([x, ff_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)

        return x

    def create_gpt2_model(
        self,
    ):
        """create_gpt2_model

        Create a GPT-2 style untrainied language model with token and positional embeddings, multiple transformer blocks, and an output projection layer.

         Arguments: none

         Returns:
             Untrained tf.Keras.model

         The model has several types of layers:

         - Input layer (input_ids)
         - Token embedding layer
         - Position embedding layer
         - Addition layer (to combine token and position embeddings)
         - Dropout layer (a certain percentage of the combined embedding values will be randomly set to zero, helping the model generalize better)
         - Multiple transformer blocks (the number is determined by self.n_layers)
         - Output dense layer (logits)

         All of these layers are trainable by default in TensorFlow/Keras. The specific number of transformer blocks depends
         on the value of self.n_layers. The transformer blocks themselves would contain multiple layers each (typically attention layers,
         normalization layers, and feedforward networks), so the total layer count would be:

         - 5 base layers (input, embeddings, add, dropout, output
         - Plus self.n_layers * (number of layers in each transformer block)

         This function creates a small GPT-2 style language model with the following parameters:

         The vocab_size parameter defines the total number of unique tokens
         that your language model can recognize and generate.
         Using our example model with d_model=256:

         Vocab Size	Embedding Parameters	Output Layer Parameters	Total Added Parameters
         5,000	    1.28M	                1.28M	                2.56M
         10,000	    2.56M	                2.56M	                5.12M
         30,000	    7.68M	                7.68M	                15.36M
         50,000	    12.8M	                12.8M	                25.6M

         The context_size parameter (also called "sequence length" or "context window")
         defines the maximum number of tokens that the model can process or generate at once.
         It represents the "memory" of the model –how much previous text the model can consider
         when generating the next token.

         The d_model parameter represents the dimensionality of the model's embedding space and is one
         of the most important hyperparameters in transformer-based architectures.

         Specifically, d_model determines:

         - The dimensionality of the token embeddings - how many features are used to represent each token
         - The dimensionality of the positional embeddings - how position information is encoded
         - The width of the attention layers - the size of the keys, queries, and values in the attention mechanism
         - The dimensionality throughout most of the model's internal representations

         In this code d_model is set to 256 by default (vector with 256 floats), which is small
         compared to larger models like GPT-2 (which uses 768 in its smallest version).
        """
        # Input tokens and combined positional embeddings
        input_ids = layers.Input(
            shape=(self.context_size,), dtype=tf.int32, name="input_ids"
        )
        x = TokenAndPositionEmbedding(
            vocab_size=self.vocab_size,
            context_size=self.context_size,
            d_model=self.d_model,
            name="token_and_position_embeddings",
        )(input_ids)

        """
        Token Embedding vs. Positional Embedding
        Both are critical components in transformer-based models, but they serve different purposes:

        Token Embedding
        Purpose: Represents the semantic meaning of each token in the vocabulary.

        Characteristics:

        - Converts tokens (words/subwords) into dense vector representations
        - Captures semantic relationships between tokens
        - Same token gets the same embedding regardless of position
        - Learned during training to encode meaning and context
        - Dimension typically ranges from 128 to 1024
        - Example: The word "bank" would have a single token embedding that tries to capture its 
          meaning, regardless of where it appears in a sentence.

        Positional Embedding
        Purpose: Encodes the position/location of each token in the sequence.

        Characteristics:

        - Provides information about token order in the sequence
        - Necessary because transformer attention mechanisms have no inherent notion of order
        - Can be learned or fixed (using mathematical functions)
        - Allows the model to understand concepts like word order, syntax, and proximity
        - Has the same dimension as token embeddings to allow addition
        - Example: The word "bank" would get a different positional embedding when it appears as the 
          1st word versus when it appears as the 5th word.

        How They Work Together In Transformer Models:

        - Each token is converted to a token embedding
        - A positional embedding corresponding to the token's position is added
        - The result is the input representation: Input = Token Embedding + Positional Embedding
        - This combined embedding allows the model to process both:

        Without positional embeddings, a transformer would treat "The dog chased the cat" and 
        "The cat chased the dog" as equivalent, since it would only see the same set of tokens 
        without position information.
        """

        x = layers.Dropout(self.dropout_rate)(x)

        # Transformer blocks
        for i in range(self.n_layers):
            x = self.transformer_block(x)

        # Output projection
        logits = layers.Dense(self.vocab_size, name="logits")(x)

        # Create model
        model = Model(inputs=input_ids, outputs=logits)
        return model

    # Create a simple tokenizer
    def configure_learning(self, decay_steps: int):
        """configure_learning

        Create and return a learning rate schedule based on training parameters.

        Arguments:
            decay_steps -- Number of steps over which to decay the learning rate

        Returns:
            PolynomialDecay learning rate schedule
        """
        lr_schedule = PolynomialDecay(
            initial_learning_rate=self.learning_rate,
            end_learning_rate=self.learning_rate / 10,
            decay_steps=decay_steps,
        )
        return lr_schedule

    def create_tokenizer(self):
        """create_tokenizer

        Create a Keras TextVectorization tokenizer and set the maximum token number and sequence length.
        Any tokens that are less frequent and fall outside the vocabulary limit will be replaced with an
        out-of-vocabulary token.

        Arguments: none

        Returns: none
        """
        if self.verbose:
            print(
                "create_tokenizer() - initialize a tokenizer (tf.keras.layers.TextVectorization) and set the sequence length"
            )

        self.tokenizer = layers.TextVectorization(
            max_tokens=self.vocab_size,
            output_mode="int",
            output_sequence_length=self.context_size,
        )
        """
        When training the modelyou use the first context_size tokens as the input and the 
        last context_size tokens as the target
        """

    def adapt(self, texts):
        """adapt

        Run adapt() to create vocabulary and make a token_id/token dictionary

        Arguments:
            texts -- list of strings

        Returns: none

        >>> import tensorflow as tf
        >>> texts = ["I love machine learning", "Machine learning is fun"]
        >>> vectorizer = tf.keras.layers.TextVectorization()
        >>> vectorizer.adapt(texts)
        >>> sequences = vectorizer(texts)
        >>> print(sequences)
        tf.Tensor(
        [[6 4 2 3]
        [2 3 5 7]], shape=(2, 4), dtype=int64)
        >>> vectorizer.get_vocabulary()
        ['', '[UNK]', 'machine', 'learning', 'love', 'is', 'i', 'fun']
        """

        self.tokenizer.adapt(texts)

        self.create_index()

        if self.verbose:
            print(f"adapt() - vocabulary size: {len(self.index_word.keys())}")

    def create_index(self):
        """create_index

        Create an index for decoding
        """
        self.index_word = {
            index: word
            for index, word in enumerate(self.tokenizer.get_vocabulary())
        }

    def load_text(self, input_dir, percentage):
        """load_text

        Read input files from a directory

        Arguments:
            input_dir -- input directory with text files
            percentage -- percentage of strings to return

        Returns:
            text - list of strings
        """
        text = ""
        if self.verbose:
            print(
                "load_text() - read input text files and return list of strings"
            )
        file_paths = glob.glob(f"{input_dir}/*")
        if percentage != 100:
            if percentage > 100:
                sys.exit("Invalid percentage: {percentage}")
            num_files = int(len(file_paths) * percentage / 100)
            file_paths = random.sample(file_paths, num_files)

        if self.verbose:
            print(
                f"load_text() - using {percentage}% of files in {input_dir} for the dataset"
            )
        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8") as f:
                text += f.read() + "\n\n"

        return text

    def analyze_text(self, chunks):
        """analyze_text

        Analyze sentence lengths and create a histogram. Create a dedicated tokenizer for analysis
        without the output_sequence_length parameter so it does not pad with 0's.

        Arguments:
            chunks -- list of strings

        """
        analysis_tokenizer = layers.TextVectorization(
            max_tokens=50000,
            output_mode="int",
        )
        analysis_tokenizer.adapt(chunks)

        token_counts = []
        chunk_lengths = []
        for chunk in chunks:
            tokens = analysis_tokenizer(chunk)
            token_counts.append(len(tokens))
            chunk_lengths.append(len(chunk))

        print(
            f"analyze_text() - mean chunk length: {np.mean(token_counts):.1f} tokens"
        )
        print(
            f"analyze_text() - median chunk length: {np.median(token_counts):.1f} tokens"
        )
        print(
            f"analyze_text() - 95th percentile: {np.percentile(token_counts, 95):.1f} tokens"
        )
        print(
            f"analyze_text() - 99th percentile: {np.percentile(token_counts, 99):.1f} tokens"
        )
        print(
            f"analyze_text() - max chunk length: {np.max(token_counts)} tokens"
        )
        print(
            f"analyze_text() - min chunk length: {np.min(token_counts)} tokens"
        )

        # Histogram
        import matplotlib.pyplot as plt

        plt.hist(token_counts, bins=30)
        plt.title(f"Distribution of {self.download} Tokens per Chunk")
        plt.xlabel("Number of Tokens")
        plt.ylabel("Frequency")
        plt.savefig("token_number_distribution.png")

        plt.hist(chunk_lengths, bins=30)
        plt.title(f"Distribution of {self.download} Chunk lengths")
        plt.xlabel("Number of Characters")
        plt.ylabel("Frequency")
        plt.savefig("chunk_length_distribution.png")

    def prepare_datasets(
        self,
        texts,
        train_size=0.8,
    ):
        """prepare_datasets

        Arguments:
            texts -- list of strings
            train_size -- float, proportion of data to use for training (default: 0.8)

        Returns:
            train_dataset - tf.data.Dataset.from_tensor_slices
            val_dataset - tf.data.Dataset.from_tensor_slices

        1. Tokenizes the input texts into a flat array of integer token IDs
        2. Creates examples by sliding a window of size context_size + 1 over the token sequence
        3. Splits each example into input (all tokens except the last) and target (all tokens except the first)
        4. Creates a TensorFlow dataset from these input/target pairs
        5. Applies shuffling, batching, and prefetching for efficient training

        """
        if self.verbose:
            print(
                "prepare_datasets() - tokenize, prepare input and target token sequences, and create tf.data.Dataset.from_tensor_slices training and validation datasets"
            )
        """Filter out arrays with excessive padding"""
        for token_array in self.tokenizer(texts):
            non_padding_count = np.sum(token_array != 0)
            if non_padding_count > 4:
                self.token_ids.append(token_array)
        """
        Create a flat array of token IDs representing all tokens from the input texts in order. 
        An alternative would be to create examples from each individual chunk.
        """
        self.token_ids = np.array(self.token_ids).flatten()

        if self.verbose:
            print(
                f"prepare_datasets() - number of tokens: {len(self.token_ids)}"
            )
        """Create examples with context_size + 1 (inputs and targets)"""
        examples = []
        for i in range(
            0, len(self.token_ids) - self.context_size, self.stride
        ):
            examples.append(self.token_ids[i : i + self.context_size + 1])
        examples = np.array(examples)

        """
        Take all tokens except the last one from each example sequence.
        For example, if we have a sequence of tokens [3, 5, 55, 4, 66]:

        The input sequence would be [3, 5, 55, 4]
        The target sequence would be [ 5, 55, 4, 66]
        """
        inputs = examples[:, :-1]  # tensor 1
        """
        When training language models, the target token is the next token that should 
        follow after seeing a sequence of input tokens.
        """
        targets = examples[:, 1:]  # tensor 2

        # Split the data into training and validation sets using sklearn
        (
            train_inputs,
            val_inputs,
            train_targets,
            val_targets,
        ) = train_test_split(
            inputs, targets, train_size=train_size, random_state=42
        )

        if self.verbose:
            print(
                f"prepare_datasets() - training samples: {len(train_inputs)}, validation samples: {len(val_inputs)}"
            )

        # Create TF datasets for training and validation
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (train_inputs, train_targets)
        )
        train_dataset = train_dataset.shuffle(10000).batch(
            self.batch_size, drop_remainder=True
        )
        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices(
            (val_inputs, val_targets)
        )
        val_dataset = val_dataset.batch(self.batch_size, drop_remainder=True)
        val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        if self.verbose:
            print(
                f"prepare_datasets() - train_dataset: {train_dataset}, val_dataset: {val_dataset}"
            )

        return train_dataset, val_dataset

    def train_model(
        self,
        train_dataset,
        val_dataset,
        model,
        checkpoint_dir="./checkpoints",
        save_checkpoint_freq=5,  # Save checkpoint every N epochs
        max_checkpoints_to_keep=3,  # Keep only N best checkpoints
        cleanup_old_checkpoints=True,  # Whether to delete old checkpoints
    ):
        """train_model

        Set up checkpoints, compile(), and run model.fit()

        Arguments:
            train_dataset -- tf.data.Dataset.from_tensor_slices
            model -- keras.src.models.functional.Functional, untrained

        Keyword Arguments:
            checkpoint_dir -- checkpoint directory (default: {"./checkpoints"})

        Returns: none

        Total Parameters

        The total number of weights and biases in the entire model. This represents the complete set of
        values that define the model's behavior and what it has learned.

        Trainable Parameters

        These are parameters that are updated during the training process through backpropagation and
        gradient descent. They're the parts of the model that "learn" from the training data. Most
        parameters in a typical neural network are trainable.

        Non-Trainable Parameters

        These parameters are not updated during training. They remain fixed at their initial values or
        at values they were set to previously. Non-trainable parameters can come from:

        - Layers that are explicitly frozen (set to non-trainable) during transfer learning
        - Batch normalization statistics (moving means and variances) that are updated during training but not
          through backpropagation
        - Embedding layers that are set to non-trainable (like when using pre-trained word embeddings)
        - Parameters in layers where training is disabled

        An epoch represents one complete pass through the entire training dataset. Within each epoch,
        the training data is processed in smaller batches, with each batch being a "step."

        Steps in Each Epoch

        - Forward Pass: For each batch, the model makes predictions based on current parameters
        - Loss Calculation: The error/loss between predictions and actual values is computed
        - Backward Pass (Backpropagation): Gradients are calculated to determine how to adjust parameters
        - Parameter Update: Weights and biases are updated according to the optimizer's rules
        - Metrics Tracking: Performance metrics are updated (accuracy, loss, etc.)
        - Repeat: Steps 1-5 are repeated for each batch until the full dataset is processed
        - Validation (optional): After all training batches, the model is evaluated on validation data

        How the Number of Steps is Determined

        The number of steps per epoch is calculated using this formula:

        steps_per_epoch = ceil(total_training_samples / batch_size)

        For example, if you have 10,000 training samples and a batch size of 32:

        Steps per epoch = ceil(10,000 / 32) = 313 steps

        Factors affecting the number of steps:

        Dataset size: Larger datasets require more steps
        Batch size: Smaller batches mean more steps per epoch
        Data handling: When using data generators or tf.data pipelines, steps may be explicitly set
        Distributed training: With multiple GPUs/TPUs, effective batch size increases, reducing steps

        In frameworks like TensorFlow/Keras, you can either:

        - Let the framework calculate steps automatically when providing a NumPy array
        - Specify steps per epoch manually when using generators or tf.data

        "Samples" in the Context of a GPT-2 Style LLM

        In the context of training GPT-2 style language models, "samples" has a specific meaning
        that differs from some other machine learning contexts:

        What Constitutes a Sample in LLM Training for GPT-2 style models:

        A sample is typically a sequence of tokens of a specific length (e.g., 512 or 1024 tokens)
        These sequences are often extracted from a larger corpus of text
        Each sample serves as a training example for the model to learn from

        Important Characteristics:

        - Sequence-based: Unlike image classification where one image = one sample, LLM samples are
          sequences of tokens
        - Context window: The sample length is determined by the model's context window (maximum sequence length)
        - Batching: Multiple samples are grouped into batches for efficient processing
        - Tokenization: Raw text must be tokenized before becoming samples

        Example

        If training a GPT-2 model with a context length of 512:

        A book might be tokenized into 50,000 tokens
        This could be divided into ~98 samples of 512 tokens each
        These samples become the training examples

        So when calculating steps per epoch:

        steps_per_epoch = ceil(number_of_sequences / batch_size)

        Where "number_of_sequences" is how many of these fixed-length token sequences you have in your training dataset.

        Learning Rate

        The current learning rate is 5e-5, but it could be reduced to 1e-5.
        Reducing the learning rate typically slows down the training process. Here's why:

        - Smaller Parameter Updates: Each training step will make smaller adjustments to the model parameters.
        - More Steps Required: The model will need more steps (and therefore more epochs) to reach the same level of performance.
        - Slower Convergence: The model will take longer to converge to an optimal solution.

        Higher learning rate: Taking large steps (might reach the destination faster but could overshoot)
        Lower learning rate: Taking small, cautious steps (takes longer but less likely to miss the target)

        The benefit of a lower learning rate is that it often provides:

        - More Stability: Less likely to encounter NaN values or divergence
        - Better Final Results: May ultimately reach a better minimum, even if it takes longer
        - Smoother Loss Curve: Less oscillation in the training/validation loss
        """

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Safely get dataset size - handle resource dtype issue
        try:
            cardinality = tf.data.experimental.cardinality(train_dataset)
            if cardinality == tf.data.experimental.UNKNOWN_CARDINALITY:
                # Estimate dataset size if cardinality is unknown
                train_dataset_size = 1000  # Default fallback value
                if self.verbose:
                    print(
                        "Warning: Dataset cardinality is unknown, using default value for decay_steps calculation"
                    )
            else:
                train_dataset_size = int(cardinality)
        except (ValueError, TypeError) as e:
            # Fallback if cardinality conversion fails
            train_dataset_size = 1000  # Default fallback value
            if self.verbose:
                print(
                    f"Warning: Could not determine dataset cardinality ({e}), using default value for decay_steps calculation"
                )

        decay_steps = self.epochs * train_dataset_size // self.batch_size

        # Create learning rate schedule
        lr_schedule = self.configure_learning(decay_steps)

        """Set up optimizer"""
        if self.optimizer == "adam":
            optimizer = tf.keras.optimizers.legacy.Adam(
                learning_rate=lr_schedule, epsilon=1e-8
            )
        elif self.optimizer == "sgd":
            optimizer = tf.keras.optimizers.legacy.SGD(
                learning_rate=self.learning_rate
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")

        """        
        Logits: when the model makes a prediction for the next token in a sequence, 
        it outputs a vector of real numbers (one for each token in your vocabulary). 
        These raw output values are called "logits". Logits are not probabilities - they can be any real number 
        (positive, negative, or zero). To convert logits to probabilities, you typically apply a softmax function.
        """
        model.compile(
            optimizer=optimizer,
            loss=SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        """ Callbacks """
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        )

        """Smart checkpoint callback that keeps disk space in check"""
        checkpoint_callback = SmartCheckpointCallback(
            checkpoint_dir=checkpoint_dir,
            save_checkpoint_freq=save_checkpoint_freq,
            max_checkpoints_to_keep=max_checkpoints_to_keep,
            cleanup_old_checkpoints=cleanup_old_checkpoints,
            verbose=self.verbose,
        )

        validation_callback = ValidationPrintCallback(val_dataset)

        callbacks = [
            early_stopping,
            checkpoint_callback,
            validation_callback,
        ]

        fit_params = {
            "validation_data": val_dataset,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "verbose": 1,
        }

        self.history = model.fit(
            train_dataset, callbacks=callbacks, **fit_params
        )

        history_metrics = self.history.history
        val_loss = history_metrics.get("val_loss", [])
        val_accuracy = history_metrics.get("val_accuracy", [])
        train_loss = history_metrics.get("loss", [])
        train_accuracy = history_metrics.get("accuracy", [])

        last_val_loss = self._scalar_metric(val_loss[-1]) if val_loss else None
        last_val_accuracy = (
            self._scalar_metric(val_accuracy[-1]) if val_accuracy else None
        )
        last_train_loss = (
            self._scalar_metric(train_loss[-1]) if train_loss else None
        )
        last_train_accuracy = (
            self._scalar_metric(train_accuracy[-1]) if train_accuracy else None
        )

        if self.verbose:
            if last_val_loss is not None:
                print(f"val_loss: {last_val_loss}")
            if last_val_accuracy is not None:
                print(f"val_accuracy: {last_val_accuracy}")
            if last_train_loss is not None:
                print(f"train_loss: {last_train_loss}")
            if last_train_accuracy is not None:
                print(f"train_accuracy: {last_train_accuracy}")

        return model

    def save(self, model):
        """save

        Save model and tokenizer separately.

        Arguments:
            model -- trained model

        Returns: none
        """
        model.save(f"{self.name}.keras")
        if self.verbose:
            print(f"save() - saved Keras model: {self.name}.keras")
        with open(f"{self.name}.pkl", "wb") as p:
            tokenizer_payload = {
                "config": self.tokenizer.get_config(),
                "vocab": self.tokenizer.get_vocabulary(),
            }
            pickle.dump(tokenizer_payload, p, protocol=pickle.HIGHEST_PROTOCOL)
        if self.verbose:
            print(f"save() - saved tokenizer: {self.name}.pkl")

    def id_to_word(self, token_id):
        """id_to_word

        Get a token given a token id

        Arguments:
            token_id -- token id

        Returns:
            Token or None
        """
        return self.index_word.get(token_id, None)

    def _scalar_metric(self, value):
        """Convert metric outputs to Python floats for stable logging."""
        if value is None:
            return None
        if isinstance(value, tf.Variable):
            value = value.read_value()
        if tf.is_tensor(value):
            value = tf.keras.backend.get_value(value)
        if isinstance(value, (np.ndarray, np.generic)):
            try:
                value = np.asarray(value).item()
            except ValueError:
                return None
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    # Function to generate text
    def generate_text(
        self,
        prompt,
        model,
        max_length: int = 100,
    ):
        """generate_text

        Generate text using the trained model

        Arguments:
            prompt: Initial text prompt to start generation
            model: Keras model

        Keyword Arguments:
            max_length: Maximum length of generated sequence

        Returns:
            Generated text as a string
        """

        """
        When an LLM encounters words/tokens not in its vocabulary during generation, several
        mechanisms come into play.

        Tokenization Breakdown

        Most modern LLMs use subword tokenization (BPE, WordPiece, SentencePiece)
        Unknown words get split into known subword units when possible
        Example: "cryptocurrency" might become ["crypto", "##curr", "##ency"]

        Special Unknown Token:

        If a word can't be broken into known subwords, it's replaced with a special
        token like <unk>, [UNK], or <unknown> depending on the tokenizer.
        """

        prompt_ids = self.tokenizer(tf.convert_to_tensor([prompt]))[0]
        if self.verbose:
            print(f"generate_text() - prompt_ids: {prompt_ids}")

        # Safely convert to numpy and reshape - handle resource tensor issue
        try:
            # Convert to numpy array safely
            if hasattr(prompt_ids, "numpy"):
                prompt_ids_array = prompt_ids.numpy()
            else:
                # If it's already a numpy array or regular tensor
                prompt_ids_array = np.array(prompt_ids)
        except (ValueError, TypeError) as e:
            if self.verbose:
                print(
                    f"Warning: Could not convert tensor to numpy ({e}), attempting alternative conversion"
                )
            # Alternative conversion method
            prompt_ids_array = tf.make_ndarray(
                tf.make_tensor_proto(prompt_ids)
            )

        # Reshape and add a batch dimension
        prompt_ids = prompt_ids_array.reshape(1, -1)

        """Generate text token by token
            
            Low temperature (0.1-0.5):

            - More deterministic, predictable outputs
            - Often more factual and coherent
            - May become repetitive or generic

            High temperature (0.7-1.5):

            - More diverse and creative outputs
            - More surprising word choices
            - May introduce more errors or nonsensical content

            Temperature = 0:

            This is a special case called "greedy decoding" where only the 
            highest probability token is selected. Always produces the same  
            output for a given prompt

            Formula: P(token_i) = softmax(logits_i / temperature)
            """
        for _ in range(max_length):
            # Get predictions from model
            predictions = model.predict(prompt_ids, verbose=0)[0][-1]

            # Apply temperature scaling
            scaled_predictions = predictions / self.temperature

            # Sample next token
            predicted_tensor = tf.random.categorical(
                tf.expand_dims(scaled_predictions, 0), num_samples=1
            )[0, 0]

            # Safely convert predicted token to integer
            try:
                predicted_id = int(predicted_tensor.numpy())
            except (ValueError, TypeError):
                # Alternative conversion if .numpy() fails
                predicted_id = int(predicted_tensor)

            # Update input context by shifting left and adding new token
            prompt_ids = np.roll(prompt_ids, -1, axis=1)
            prompt_ids[0, -1] = predicted_id

            # Convert token to word and append to result
            word = self.id_to_word(predicted_id)
            if word:
                # Stop if we generate an end token
                if word.lower() in {"<eos>", "eos", "[eos]"}:
                    prompt += "."
                    break
                prompt += " " + word
            elif self.verbose:
                print(f"generate_text() - no token for id {predicted_id}")

        return prompt

    def clean_cc_news(self, datasets, min_chunk_len):
        """clean_cc_news
        Clean and filter CC-News text data by extracting high-quality text chunks.

        This method processes CC-News datasets by splitting texts on newlines and
        filtering out chunks that don't meet a minimum alphabetic character threshold.
        Only text chunks with more than 70% alphabetic characters are retained.

        Also filters out chunks that are shorter than a specified minimum length.
        These short chunks are likely not complete sentences, e.g.:

        ['Literary Roots', 'From Page to Stage', 'Out of Russia', 'Across the Country…', 'A Christmas Staple', 'Nutcracker All Over',
        'Getting In', 'A Typical Day', 'Stage Time', 'Share Us', 'Share Us', 'Installation', 'Configure Apache', 'Accessing Trac', 'Figure A',
        'Congratulations', 'Also see', 'Also see', 'Setup', 'Figure A', 'Language selection', 'EULA', ...]

        Args:
            datasets (datasets.arrow_dataset.Dataset): List of dictionaries containing text data, where each
            dictionary has a "text" key with the content to be processed.

        Returns:
            list: A list of cleaned text strings that passed the alphabetic character
                threshold filter.
        """
        chunks = list()
        if self.verbose:
            print(
                f"clean_cc_news() - number of cc_news datasets: {len(datasets)}"
            )
        for dataset in datasets:
            subtxts = dataset["text"].split("\n")
            for subtxt in subtxts:
                alpha_count = sum(1 for char in subtxt if char.isalpha())
                if (alpha_count / len(subtxt)) > 0.7 and len(
                    subtxt
                ) > min_chunk_len:
                    """Periods are just normal punctuation tokens, not the special <EOS> string.
                    TextVectorization doesn’t insert <EOS> automatically, so we do that here
                    to mark the end of each sentence in the chunk."""
                    chunks.append(f"{subtxt.replace('.', '. <EOS>')}")
        if self.verbose:
            print(f"clean_cc_news() - number of text chunks: {len(chunks)}")
        return chunks

    def clean_wikitext(self, raw_texts, percentage):
        """clean_wikitext

        Clean and preprocess WikiText dataset sentences.

        This method processes raw WikiText data by filtering out headers, removing
        sentences with unknown tokens and URLs, fixing tokenization artifacts, and
        optionally sampling a percentage of the cleaned sentences.

        Args:
            raw_texts: A dataset dictionary containing a "train" split with text entries.
                    Each entry should have a "text" field containing the raw text.
            percentage: Float or int between 0-100 indicating what percentage of cleaned
                    sentences to retain. If 100, all cleaned sentences are kept.

        Returns:
            List[str]: A list of cleaned sentences ready for use in training/evaluation.
                    Sentences have been filtered, cleaned, and potentially sampled.

        Example:
            >>> slam = SLAM()
            >>> raw_data = {"train": [{"text": "This is a sentence."}, {"text": "= Header ="}]}
            >>> cleaned = slam.clean_wikitext(raw_data, 50)  # Keep 50% of sentences
            >>> # Returns: ['This is a sentence.'] (header filtered out)

        Note:
            The cleaning process:
            1. Removes lines starting with "=" (WikiText headers)
            2. Splits text into sentences using NLTK
            3. Filters out sentences containing "<unk>" tokens or "http" URLs
            4. Fixes NLTK tokenization spacing issues around punctuation
            5. Randomly samples sentences if percentage < 100
        """
        sentences = list()
        texts = [t["text"].strip() for t in raw_texts["train"]]
        texts = [t for t in texts if not t.startswith("=")]
        for text in texts:
            for sentence in nltk.sent_tokenize(text):
                if "<unk>" not in sentence and "http" not in sentence:
                    # The nltk tokenizer introduces spaces
                    sentence = re.sub(r"\s+([.,?!:;'])", r"\1", sentence)
                    sentences.append(f"{sentence} <EOS>")
        if self.verbose:
            print(
                f"clean_wikitext() - number of cleaned sentences: {len(sentences)}"
            )
        if percentage != 100:
            num_sentences = int(len(sentences) * percentage / 100)
            sentences = random.sample(sentences, num_sentences)
        if self.verbose:
            print(
                f"clean_wikitext() - using {percentage}% ({len(sentences)}) of the cleaned sentences for the datasets"
            )
        """For example: 'As a liquid , xenon has a density of up to 3 @.' """
        return sentences

    def load(self, name):
        """load
        Load a previously saved tokenizer and model from disk.

        Loads the tokenizer from a pickle file and the Keras model from a .keras file,
        then recreates the token index for the loaded tokenizer.

        Args:
            name (str): The base name for the saved files (without extensions).
                       Will look for {name}.pickle and {name}.keras files.

        Returns:
            tf.keras.Model: The loaded Keras model.

        Raises:
            SystemExit: If either the tokenizer pickle file or model file is not found.

        Example:
            model = slam_instance.load("my_saved_model")
        """
        self.name = name
        if not os.path.exists(f"{self.name}.pkl"):
            sys.exit(f"Tokenizer pickle file not found: {self.name}.pkl")
        with open(f"{self.name}.pkl", "rb") as f:
            tokenizer_payload = pickle.load(f)
            if (
                isinstance(tokenizer_payload, dict)
                and "config" in tokenizer_payload
                and "vocab" in tokenizer_payload
            ):
                self.tokenizer = layers.TextVectorization.from_config(
                    tokenizer_payload["config"]
                )
                self.tokenizer.set_vocabulary(tokenizer_payload["vocab"])
            else:
                self.tokenizer = tokenizer_payload
        self.create_index()

        if not os.path.exists(f"{self.name}.keras"):
            sys.exit(f"Model file not found: {self.name}.keras")
        try:
            model = tf.keras.models.load_model(f"{self.name}.keras")
        except Exception as e:
            sys.exit(
                f"Model file could not be loaded with the current architecture. Error: {e}"
            )
        return model


class ValidationPrintCallback(tf.keras.callbacks.Callback):
    """ValidationPrintCallback

    A custom Keras callback that prints validation metrics at the end of each epoch.

    This callback evaluates the model on validation data and prints the results
    in a formatted manner, making it easier to track validation performance
    during training.

    Attributes:
        validation_data: The validation dataset to evaluate the model on.

    Args:
        validation_data: A tf.data.Dataset or tuple of (inputs, targets)
                        containing the validation data.
    """

    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        """on_epoch_end

        Called at the end of each epoch to evaluate and print validation metrics.

        Args:
            epoch (int): The current epoch number.
            logs (dict, optional): Dictionary of logs from the training epoch.
                                  Not used in this implementation.
        """
        val_results = self.model.evaluate(self.validation_data, verbose=0)
        metric_names = self.model.metrics_names
        print(f"\nValidation results {epoch}:")
        for name, value in zip(metric_names, val_results):
            print(f"val_{name}: {value:.4f}")


class SmartCheckpointCallback(tf.keras.callbacks.Callback):
    """SmartCheckpointCallback

    A space-efficient checkpoint callback that manages disk usage by:
    1. Saving checkpoints only at specified intervals (not every epoch)
    2. Keeping only the N best checkpoints based on validation loss
    3. Automatically cleaning up old/worse checkpoints
    4. Providing detailed logging about checkpoint management

    This callback is designed to prevent disk space issues during long training runs
    with large models that generate huge checkpoint files.
    """

    def __init__(
        self,
        checkpoint_dir="./checkpoints",
        save_checkpoint_freq=5,
        max_checkpoints_to_keep=3,
        cleanup_old_checkpoints=True,
        verbose=True,
        monitor="val_loss",
        mode="min",
    ):
        """Initialize the SmartCheckpointCallback.

        Args:
            checkpoint_dir (str): Directory to save checkpoints
            save_checkpoint_freq (int): Save checkpoint every N epochs
            max_checkpoints_to_keep (int): Maximum number of checkpoints to keep
            cleanup_old_checkpoints (bool): Whether to delete old checkpoints
            verbose (bool): Whether to print checkpoint management messages
            monitor (str): Metric to monitor for keeping best checkpoints
            mode (str): "min" or "max" - whether lower or higher metric values are better
        """
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.save_checkpoint_freq = save_checkpoint_freq
        self.max_checkpoints_to_keep = max_checkpoints_to_keep
        self.cleanup_old_checkpoints = cleanup_old_checkpoints
        self.verbose = verbose
        self.monitor = monitor
        self.mode = mode

        # Track best checkpoints: [(epoch, metric_value, filepath), ...]
        self.best_checkpoints = []

        # Ensure checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if self.verbose:
            print("SmartCheckpointCallback initialized:")
            print(f"  - Save frequency: every {save_checkpoint_freq} epochs")
            print(f"  - Max checkpoints to keep: {max_checkpoints_to_keep}")
            print(
                f"  - Monitoring: {monitor} ({'minimize' if mode == 'min' else 'maximize'})"
            )
            print(f"  - Cleanup enabled: {cleanup_old_checkpoints}")

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch to potentially save checkpoints."""
        logs = logs or {}

        # Only save checkpoint if it's time (based on frequency)
        if (epoch + 1) % self.save_checkpoint_freq != 0:
            return

        # Get the monitored metric value
        metric_value = logs.get(self.monitor)
        if metric_value is None:
            if self.verbose:
                print(
                    f"Warning: Monitored metric '{self.monitor}' not found in logs. Skipping checkpoint save."
                )
            return

        # Create checkpoint filepath
        checkpoint_name = f"ckpt_epoch_{epoch+1:03d}_{self.monitor}_{metric_value:.4f}.weights.h5"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)

        # Save the checkpoint
        try:
            self.model.save_weights(checkpoint_path)

            # Get file size for reporting
            file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)

            if self.verbose:
                print(
                    f"Saved checkpoint: {checkpoint_name} ({file_size_mb:.1f} MB)"
                )

            # Add to our tracking list
            self.best_checkpoints.append(
                (epoch + 1, metric_value, checkpoint_path)
            )

            # Manage checkpoint storage
            if self.cleanup_old_checkpoints:
                self._cleanup_checkpoints()

        except Exception as e:
            if self.verbose:
                print(f"Error saving checkpoint: {e}")

    def _cleanup_checkpoints(self):
        """Remove excess checkpoints, keeping only the best ones."""
        if len(self.best_checkpoints) <= self.max_checkpoints_to_keep:
            return

        # Sort by metric value (best first)
        if self.mode == "min":
            self.best_checkpoints.sort(
                key=lambda x: x[1]
            )  # Sort by metric value ascending
        else:
            self.best_checkpoints.sort(
                key=lambda x: x[1], reverse=True
            )  # Sort by metric value descending

        # Keep only the best checkpoints
        checkpoints_to_keep = self.best_checkpoints[
            : self.max_checkpoints_to_keep
        ]
        checkpoints_to_remove = self.best_checkpoints[
            self.max_checkpoints_to_keep :
        ]

        # Remove excess checkpoint files
        total_freed_mb = 0
        for epoch, metric_val, filepath in checkpoints_to_remove:
            try:
                if os.path.exists(filepath):
                    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
                    os.remove(filepath)
                    total_freed_mb += file_size_mb
                    if self.verbose:
                        print(
                            f"Removed checkpoint: {os.path.basename(filepath)} (freed {file_size_mb:.1f} MB)"
                        )
            except Exception as e:
                if self.verbose:
                    print(f"Error removing checkpoint {filepath}: {e}")

        # Update our tracking list
        self.best_checkpoints = checkpoints_to_keep

        if self.verbose and total_freed_mb > 0:
            print(f"Total disk space freed: {total_freed_mb:.1f} MB")

            # Show current checkpoint status
            print(f"Keeping {len(checkpoints_to_keep)} best checkpoints:")
            for epoch, metric_val, filepath in checkpoints_to_keep:
                file_size_mb = (
                    os.path.getsize(filepath) / (1024 * 1024)
                    if os.path.exists(filepath)
                    else 0
                )
                print(
                    f"  - Epoch {epoch}: {self.monitor}={metric_val:.4f} ({file_size_mb:.1f} MB)"
                )

    def get_best_checkpoint(self):
        """Return the path to the best checkpoint."""
        if not self.best_checkpoints:
            return None

        # Sort to get the best one
        if self.mode == "min":
            best_checkpoint = min(self.best_checkpoints, key=lambda x: x[1])
        else:
            best_checkpoint = max(self.best_checkpoints, key=lambda x: x[1])

        return best_checkpoint[2]  # Return filepath
