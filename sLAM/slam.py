import tensorflow as tf
import numpy as np
import glob
import os
import random
import sys
import time
import pickle
import nltk
from tensorflow.keras import layers  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.optimizers.schedules import PolynomialDecay  # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint  # type: ignore  # noqa: F401


class slam_builder:
    """
    A simple language model using transformer decoder-only architecture.
    Hyperparameters are set here before training (parameters are learned in the training,
    e.g. weights, values in the embedding matrices, coefficients in linear regression).
    """

    def __init__(
        self,
        verbose: bool = False,
        name: str = None,
        vocab_size: int = 50000,
        context_size: int = 256,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int = 1024,
        dropout_rate: float = 0.1,
        epochs: int = 1,
        batch_size: int = 4,
    ):
        """__init__

        Keyword Arguments:
            verbose -- Print out details (default: {False})
            name -- Name identifier for the model (default: {None})
            vocab_size -- Size of the vocabulary/token dictionary (default: {50000})
            context_size -- Maximum sequence length for input contexts (default: {256})
            d_model -- Dimensionality of the model's embeddings (default: {256})
            n_layers -- Number of transformer layers in the model (default: {4})
            n_heads -- Number of attention heads in each transformer layer (default: {4})
            d_ff -- Dimensionality of the feed-forward network (default: {1024})
            dropout_rate -- Rate of dropout for regularization (default: {0.1})
            epochs -- Number of training epochs (default: {1})
            batch_size -- Number of samples per training batch (default: {4})

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

        """
        Mixed precision training
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        """

        # Set memory growth to avoid OOM issues
        physical_devices = tf.config.list_physical_devices("GPU")
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Define the transformer block
    def transformer_block(self, x, n_heads, d_model, d_ff, dropout_rate):
        """transformer_block

        A standard transformer block with multi-head attention and feed-forward network.

        Arguments:
            x -- Input tensor to the transformer block
            n_heads -- Number of attention heads to use
            d_model -- Dimension of the model/embedding
            d_ff -- Dimension of the feedforward network
            dropout_rate -- Rate for dropout regularization

        Returns:
            x -- Transformed tensor with the same shape as the input but processed through
                 self-attention and feed-forward layers
        """
        # Multi-head attention
        attn_output = layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=d_model // n_heads, dropout=dropout_rate
        )(x, x, x, use_causal_mask=True)

        # Residual connection and layer norm
        x = layers.Add()([x, attn_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)

        # Feed forward network
        ff_output = layers.Dense(d_ff, activation="gelu")(x)
        ff_output = layers.Dense(d_model)(ff_output)
        ff_output = layers.Dropout(dropout_rate)(ff_output)

        # Second residual connection and layer norm
        x = layers.Add()([x, ff_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)

        return x

    # Create the custom GPT-2 small model
    def create_small_gpt2_model(
        self,
    ):
        """create_small_gpt2_model

        Arguments: none

        Returns:
            Untrained tf.Keras.model

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
        # Input tokens and positional embeddings
        input_ids = layers.Input(
            shape=(self.context_size,), dtype=tf.int32, name="input_ids"
        )

        # Embedding layer
        token_embeddings = layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.d_model,
            name="token_embeddings",
        )(input_ids)

        # Positional embeddings
        positions = tf.range(start=0, limit=self.context_size, delta=1)
        position_embeddings = layers.Embedding(
            input_dim=self.context_size,
            output_dim=self.d_model,
            name="position_embeddings",
        )(positions)

        """
        Token Embedding vs. Positional Embedding
        Both are critical components in transformer-based models, but they serve different purposes:

        Token Embedding
        Purpose: Represents the semantic meaning of each token in the vocabulary.

        Characteristics:

        Converts tokens (words/subwords) into dense vector representations
        Captures semantic relationships between tokens
        Same token gets the same embedding regardless of position
        Learned during training to encode meaning and context
        Dimension typically ranges from 128 to 1024
        Example: The word "bank" would have a single token embedding that tries to capture its 
        meaning, regardless of where it appears in a sentence.

        Positional Embedding
        Purpose: Encodes the position/location of each token in the sequence.

        Characteristics:

        Provides information about token order in the sequence
        Necessary because transformer attention mechanisms have no inherent notion of order
        Can be learned or fixed (using mathematical functions)
        Allows the model to understand concepts like word order, syntax, and proximity
        Has the same dimension as token embeddings to allow addition
        Example: The word "bank" would get a different positional embedding when it appears as the 
        1st word versus when it appears as the 5th word.

        How They Work Together In Transformer Models:

        Each token is converted to a token embedding
        A positional embedding corresponding to the token's position is added
        The result is the input representation: Input = Token Embedding + Positional Embedding
        This combined embedding allows the model to process both:

        What the token means (token embedding)
        Where the token is located (positional embedding)
        Without positional embeddings, a transformer would treat "The dog chased the cat" and 
        "The cat chased the dog" as equivalent, since it would only see the same set of tokens 
        without position information.
        """

        # Make positional encodings broadcastable to batch dimension
        position_embeddings = tf.expand_dims(
            position_embeddings, axis=0
        )  # Shape: [1, seq_len, d_model]
        # Now this should work with token embeddings of shape [batch_size, seq_len, d_model]

        # Add token and position embeddings
        x = layers.Add()([token_embeddings, position_embeddings])
        x = layers.Dropout(self.dropout_rate)(x)

        # Transformer blocks
        for i in range(self.n_layers):
            x = self.transformer_block(
                x, self.n_heads, self.d_model, self.d_ff, self.dropout_rate
            )

        # Output projection
        logits = layers.Dense(self.vocab_size, name="logits")(x)

        # Create model
        model = tf.keras.Model(inputs=input_ids, outputs=logits)
        return model

    # Create a simple tokenizer
    def create_tokenizer(self):
        """create_tokenizer

        Creates a TextVectorization tokenizer and sets the maximum vocabulary size and sequence length.
        Any tokens that are less frequent and fall outside the vocabulary limit will be replaced with an
        out-of-vocabulary token.

        Arguments: none

        Returns: none
        """
        if self.verbose:
            print(
                "create_tokenizer() - initialize a tokenizer (tf.keras.layers.TextVectorization) and set the sequence length"
            )

        self.tokenizer = tf.keras.layers.TextVectorization(
            max_tokens=50000,
            output_mode="int",
            output_sequence_length=self.context_size + 1,
        )
        """
        The +1 tells the tokenizer to include the target token in the sequence. When training the model, 
        you use the first context_size tokens as the input and the last context_size tokens as the target
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

    def prepare_datasets(self, texts):
        """prepare_datasets

        1. Tokenizes the input texts into a flat array of integer token IDs
        2. Creates examples by sliding a window of size context_size + 1 over the token sequence
        3. Splits each example into input (all tokens except the last) and target (all tokens except the first)
        4. Creates a TensorFlow dataset from these input/target pairs
        5. Applies shuffling, batching, and prefetching for efficient training

        Arguments:
            texts -- list of strings

        Returns:
            dataset - tf.data.Dataset.from_tensor_slices

        Converts text to sequences of integers (token
        ids) that correspond to the indices in index_word.
        """
        if self.verbose:
            print(
                "prepare_datasets() - tokenize, prepare input and target token sequences, and create a tf.data.Dataset.from_tensor_slices dataset"
            )
        """
        Create a flat array of token IDs representing all tokens from the input texts in order. 
        """
        self.token_ids = self.tokenizer(texts).numpy().flatten()
        self.num_tokens = len(self.token_ids)

        if self.verbose:
            print(f"prepare_datasets() - number of tokens: {self.num_tokens}")
        """Create examples with context_size + 1 (inputs and targets)"""
        examples = []
        for i in range(0, len(self.token_ids) - self.context_size):
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

        # Create TF dataset
        dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
        dataset = dataset.shuffle(10000).batch(
            self.batch_size, drop_remainder=True
        )
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        """
        <_PrefetchDataset element_spec=(TensorSpec(shape=(4, 256), dtype=tf.int64, 
        name=None), TensorSpec(shape=(4, 256), dtype=tf.int64, name=None))>
        """
        if self.verbose:
            print(f"prepare_datasets() - dataset is {dataset}")

        return dataset

    # Custom training function with callbacks
    def train_model(
        self,
        train_dataset,
        model,
        learning_rate=5e-5,
        checkpoint_dir="./checkpoints",
    ):
        """train_model

        Set up optimizer, checkpoints, compile(), and run model.fit()

        Arguments:
            train_dataset -- tf.data.Dataset.from_tensor_slices
            model -- keras.src.models.functional.Functional, untrained

        Keyword Arguments:
            learning_rate -- learning rate (default: {5e-5})
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

        Layers that are explicitly frozen (set to non-trainable) during transfer learning

        Batch normalization statistics (moving means and variances) that are updated during training but not through backpropagation
        Embedding layers that are set to non-trainable (like when using pre-trained word embeddings)
        Parameters in layers where training is disabled

        An epoch represents one complete pass through the entire training dataset. Within each epoch,
        the training data is processed in smaller batches, with each batch being a "step."

        Steps in Each Epoch

        Forward Pass: For each batch, the model makes predictions based on current parameters
        Loss Calculation: The error/loss between predictions and actual values is computed
        Backward Pass (Backpropagation): Gradients are calculated to determine how to adjust parameters
        Parameter Update: Weights and biases are updated according to the optimizer's rules
        Metrics Tracking: Performance metrics are updated (accuracy, loss, etc.)
        Repeat: Steps 1-5 are repeated for each batch until the full dataset is processed
        Validation (optional): After all training batches, the model is evaluated on validation data

        How the Number of Steps is Determined

        The number of steps per epoch is calculated using this formula:

        steps_per_epoch = ceil(total_training_samples / batch_size)

        For example:

        If you have 10,000 training samples and a batch size of 32
        Steps per epoch = ceil(10,000 / 32) = 313 steps
        Factors affecting the number of steps:

        Dataset size: Larger datasets require more steps
        Batch size: Smaller batches mean more steps per epoch
        Data handling: When using data generators or tf.data pipelines, steps may be explicitly set
        Distributed training: With multiple GPUs/TPUs, effective batch size increases, reducing steps
        In frameworks like TensorFlow/Keras, you can either:

        Let the framework calculate steps automatically when providing a NumPy array
        Specify steps_per_epoch manually when using generators or tf.data

        "Samples" in the Context of a GPT-2 Style LLM

        In the context of training GPT-2 style language models, "samples" has a specific meaning
        that differs from some other machine learning contexts:

        What Constitutes a Sample in LLM Training

        For GPT-2 style models:

        A sample is typically a sequence of tokens of a specific length (e.g., 512 or 1024 tokens)
        These sequences are often extracted from a larger corpus of text
        Each sample serves as a training example for the model to learn from

        Important Characteristics

        Sequence-based: Unlike image classification where one image = one sample, LLM samples are sequences of tokens

        Context window: The sample length is determined by the model's context window (maximum sequence length)

        Batching: Multiple samples are grouped into batches for efficient processing

        Tokenization: Raw text must be tokenized before becoming samples

        Example

        If training a GPT-2 model with a context length of 512:

        A book might be tokenized into 50,000 tokens
        This could be divided into ~98 samples of 512 tokens each
        These samples become the training examples
        So when calculating steps per epoch:

        steps_per_epoch = ceil(number_of_sequences / batch_size)

        Where "number_of_sequences" is how many of these fixed-length token sequences you have in your training dataset.
        """

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

        """Learning rate schedule"""
        lr_schedule = PolynomialDecay(
            initial_learning_rate=learning_rate,
            end_learning_rate=learning_rate / 10,
            decay_steps=self.epochs * len(train_dataset),
        )

        optimizer = Adam(learning_rate=lr_schedule, epsilon=1e-8)

        checkpoint_prefix = os.path.join(
            checkpoint_dir, "ckpt_{epoch}.weights.h5"
        )
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix, save_weights_only=True
        )

        """        
        Logits in neural networks: When your model makes a prediction for the next token in a sequence, 
        it outputs a vector of real numbers (one for each token in your vocabulary). These raw output values are called "logits".

        Relationship to probabilities: Logits are not probabilities - they can be any real number (positive, negative, or zero). 
        To convert logits to probabilities, you typically apply a softmax function.
        """
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            ),
            metrics=["accuracy"],
        )

        # Train model
        self.history = model.fit(
            train_dataset, epochs=self.epochs, callbacks=[checkpoint_callback]
        )
        return model

    def save(self, model):
        """save

        Save model and tokenizer. Use a timestamp if no name is supplied.

        Arguments:
            model -- trained model

        Returns: none
        """
        if not self.name:
            self.name = time.strftime("%m-%d-%Y-%H-%M-%S", time.localtime())
        """In Tensorflow the tokenizer is usually not saved with the model, they must be saved separately"""
        model.save(f"{self.name}.keras")
        if self.verbose:
            print(f"save() - saved Keras model ({self.name}.keras)")
        with open(f"{self.name}.pickle", "wb") as p:
            pickle.dump(self.tokenizer, p, protocol=pickle.HIGHEST_PROTOCOL)
        if self.verbose:
            print(f"save() - saved tokenizer ({self.name}.pickle)")

    def id_to_word(self, token_id):
        """id_to_word

        Get a token given a token id

        Arguments:
            token_id -- token id

        Returns:
            Token or None
        """
        return self.index_word.get(token_id, None)

    def analyze_text(self, sentences):
        """analyze_text

        Analyze sentence lengths and create a histogram. Create a dedicated tokenizer for analysis
        without the output_sequence_length parameter so it does not pad with 0's.

        Arguments:
            sentences -- list of strings

        """
        analysis_tokenizer = tf.keras.layers.TextVectorization(
            max_tokens=50000,
            output_mode="int",
        )
        analysis_tokenizer.adapt(sentences)

        token_counts = []
        for sentence in sentences:
            tokens = analysis_tokenizer(sentence)
            token_counts.append(len(tokens))

        print(
            f"analyze_text() - mean sentence length: {np.mean(token_counts):.1f} tokens"
        )
        print(
            f"analyze_text() - median sentence length: {np.median(token_counts):.1f} tokens"
        )
        print(
            f"analyze_text() - 95th percentile: {np.percentile(token_counts, 95):.1f} tokens"
        )
        print(
            f"analyze_text() - 99th percentile: {np.percentile(token_counts, 99):.1f} tokens"
        )
        print(
            f"analyze_text() - max sentence length: {np.max(token_counts)} tokens"
        )

        # Histogram
        import matplotlib.pyplot as plt

        plt.hist(token_counts, bins=30)
        plt.title("Distribution of Wikipedia Sentence Lengths")
        plt.xlabel("Number of Tokens")
        plt.ylabel("Frequency")
        plt.savefig("sentence_length_distribution.png")

    # Function to generate text
    def generate_text(
        self,
        prompt,
        model,
        max_length: int = 100,
        temperature=None,
    ):
        """generate_text

        Generate text using the trained model

        Arguments:
            prompt: Initial text prompt to start generation
            model: Keras model

        Keyword Arguments:
            max_length: Maximum length of generated sequence
            temperature: Controls randomness in generation

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

        prompt_ids = self.tokenizer(tf.convert_to_tensor([prompt]))
        if self.verbose:
            print(f"generate_text() - prompt_ids: {prompt_ids}")

        # Truncate or pad if necessary
        # self.context_size = model.inputs[0].shape[1]
        prompt_ids = prompt_ids[0]
        if len(prompt_ids) > self.context_size:
            prompt_ids = prompt_ids[-self.context_size :]
        else:
            prompt_ids = [0] * (
                self.context_size - len(prompt_ids)
            ) + prompt_ids

        prompt_ids = np.array(prompt_ids)
        prompt_ids = prompt_ids.reshape(1, -1)  # Add batch dimension

        """Generate text token by token"""
        for _ in range(max_length):
            predictions = model.predict(prompt_ids, verbose=0)[0]

            # Get the predictions for the last token
            """ 
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
            predictions = predictions[-1] / temperature
            predicted_id = tf.random.categorical(
                tf.expand_dims(predictions, 0), num_samples=1
            )[-1, 0].numpy()

            # Update the input ids
            prompt_ids = np.roll(prompt_ids, -1, axis=1)
            prompt_ids[0, -1] = predicted_id

            word = self.id_to_word(predicted_id)
            if word:
                prompt += " " + word
            else:
                if self.verbose:
                    print(f"generate_text() - no token for id {predicted_id}")

            # Stop if we generate an end token
            if word == "<EOS>":
                break

        return prompt

    def clean_wikitext(self, raw_texts, percentage, min_sentence_len):
        sentences = list()
        texts = [t["text"].strip() for t in raw_texts["train"]]
        texts = [t for t in texts if not t.startswith("=")]
        for text in texts:
            for sentence in nltk.sent_tokenize(text):
                if (
                    "<unk>" not in sentence
                    and "http" not in sentence
                    and len(sentence) > min_sentence_len
                ):
                    sentences.append(sentence)
        if self.verbose:
            print(
                f"clean_wikitext() - total number of cleaned sentences: {len(sentences)})"
            )
        if percentage != 100:
            num_sentences = int(len(sentences) * percentage / 100)
            sentences = random.sample(sentences, num_sentences)
        if self.verbose:
            print(
                f"clean_wikitext() - using {percentage}% ({len(sentences)}) of the cleaned sentences for the dataset"
            )
        """For example: 'As a liquid , xenon has a density of up to 3 @.' """
        return sentences

    def load(self, name):
        self.name = name
        if not os.path.exists(f"{self.name}.pickle"):
            sys.exit(f"Tokenizer pickle file not found: {self.name}.pickle")
        with open(f"{self.name}.pickle", "rb") as f:
            self.tokenizer = pickle.load(f)
        self.create_index()

        if not os.path.exists(f"{self.name}.keras"):
            sys.exit(f"Model file not found: {self.name}.keras")
        model = tf.keras.models.load_model(f"{self.name}.keras")
        return model
