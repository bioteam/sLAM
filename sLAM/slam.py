import tensorflow as tf
import numpy as np
import os
import random
import sys
from tensorflow.keras import layers  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.optimizers.schedules import PolynomialDecay  # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint  # type: ignore  # noqa: F401

# import tensorflow_text as tf_text  # type: ignore


class slam_builder:
    """
    A simple language model based on transformer encoder/decoder architecture.
    Hyperparameters are set here before training (parameters are learned in the training,
    e.g. weights, values in the embedding matrices, coefficients in linear regression).
    """

    def __init__(
        self,
        verbose: bool = False,
        vocab_size: int = 10000,
        context_size: int = 256,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int = 1024,
        dropout_rate: float = 0.1,
        epochs: int = 3,
        batch_size: int = 4,
    ):
        self.verbose = verbose
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
        Even smaller model for extremely limited resources:
        vocab_size=5000,       # Smaller vocabulary
        context_size=128,      # Shorter context
        d_model=128,           # Smaller embeddings
        n_layers=2,            # Fewer layers
        n_heads=2,             # Fewer attention heads
        d_ff=512               # Smaller feed-forward
        """

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
        d_ff=1024,
        dropout_rate=0.1,
    ):
        """create_small_gpt2_model

        The vocab_size parameter defines the total number of unique tokens
        that your language model can recognize and generate.
        Using our example model with d_model=256:

        Vocab Size	Embedding Parameters	Output Layer Parameters	Total Added Parameters
        5,000	    1.28M	                1.28M	                2.56M
        10,000	    2.56M	                2.56M	                5.12M
        30,000	    7.68M	                7.68M	                15.36M
        50,000	    12.8M	                12.8M	                25.6M

        The context_size parameter (also commonly called "sequence length" or "context window")
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
        - In the provided code, d_model is set to 256 by default (vector with 256 floats), which is small
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

        # Make positional encodings broadcastable to batch dimension
        position_embeddings = tf.expand_dims(
            position_embeddings, axis=0
        )  # Shape: [1, seq_len, d_model]
        # Now this should work with token embeddings of shape [batch_size, seq_len, d_model]

        # Add token and position embeddings
        x = layers.Add()([token_embeddings, position_embeddings])
        x = layers.Dropout(dropout_rate)(x)

        # Transformer blocks
        for i in range(self.n_layers):
            x = self.transformer_block(
                x, self.n_heads, self.d_model, d_ff, dropout_rate
            )

        # Output projection
        logits = layers.Dense(self.vocab_size, name="logits")(x)

        # Create model
        model = tf.keras.Model(inputs=input_ids, outputs=logits)
        return model

    # Create a simple tokenizer
    def create_simple_tokenizer(self):
        """Creates a simple WordPiece tokenizer."""
        if self.verbose:
            print(
                "Create a simple tokenizer (tf.keras.preprocessing.text.Tokenizer) with an estimated vocabulary size"
            )
        vocab_size = min(self.vocab_size, 30000)  # Limit vocab for efficiency

        # bert_tokenizer = tf_text.BertTokenizer(
        #     vocab_lookup_table=None,
        #     suffix_indicator="##",
        #     max_bytes_per_word=100,
        #     max_chars_per_token=None,
        #     token_out_type=tf.int64,
        #     unknown_token="[UNK]",
        #     split_unknown_characters=False,
        # )

        # A simpler tokenizer:
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=vocab_size, oov_token="<UNK>"
        )

    def fit(self, texts):
        """
        fit_on_texts():
        1. It scans through all the texts you provide
        2. It builds a word index dictionary (mapping words to integers)
        3. It computes word frequencies and other metadata
        4. It doesn't return tokenized sequences - it just prepares the tokenizer
        """
        if self.verbose:
            print(
                "Running fit() to create word-to-int and int-to-word indices based on input text"
            )
        self.tokenizer.fit_on_texts(texts)

        # Add special tokens
        word_index = self.tokenizer.word_index
        # Padding token
        word_index["<PAD>"] = 0
        # Beginning of sequence
        word_index["<BOS>"] = len(self.word_index) + 1
        # End of sequence
        word_index["<EOS>"] = len(self.word_index) + 1

        # Reverse the word index for decoding
        self.index_word = {v: k for k, v in word_index.items()}

        # Adjust vocab_size to actual vocabulary size
        self.vocab_size = len(self.word_index) + 1
        if self.verbose:
            print(f"Actual vocabulary size: {self.vocab_size}")

    def load_text(self, file_paths):
        if self.verbose:
            print(
                "Running load_text() to read text files and return one string"
            )
        all_text = ""
        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8") as f:
                all_text += f.read() + "\n\n"
        return all_text

    # Function to prepare the dataset

    def prepare_dataset(self, text, percentage):
        """
        Converts text to sequences of integers which are token
        ids that correspond to the indices in word_index.
        """
        if self.verbose:
            print(
                "Running prepare_dataset() to tokenize, prepare input and target tokens, and create a tf.data.Dataset.from_tensor_slices"
            )
        if percentage != 100:
            if percentage > 100:
                sys.exit("Invalid percentage: {percentage}")
            if self.verbose:
                print(
                    f"Using {percentage} percent of input text in the dataset"
                )
            text_array = text.split("\n\n")
            num_elements = int(len(text_array) * percentage / 100)
            random_selection = random.sample(text_array, num_elements)
            text = "\n\n".join(random_selection)

        token_ids = self.tokenizer.texts_to_sequences([text])[0]
        # Create examples with context_size + 1 (inputs and targets)
        examples = []
        for i in range(0, len(token_ids) - self.context_size):
            examples.append(token_ids[i : i + self.context_size + 1])
        examples = np.array(examples)

        """
        Take all tokens except the last one from each example sequence.
        For example, if we have a sequence of tokens [A, B, C, D, E]:

        The input would be [A, B, C, D]
        The target would be [B, C, D, E]
        """
        inputs = examples[:, :-1]  # tensor 1
        """
        When training language models, the target is the next token that should follow 
        after seeing a sequence of input tokens.
        """
        targets = examples[:, 1:]  # tensor 2

        # Create TF dataset
        dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
        dataset = dataset.shuffle(10000).batch(
            self.batch_size, drop_remainder=True
        )
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset

    # Custom training function with callbacks
    def train_model(
        self,
        train_dataset,
        model,
        learning_rate=5e-5,
        checkpoint_dir="./checkpoints",
    ):
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Learning rate schedule
        lr_schedule = PolynomialDecay(
            initial_learning_rate=learning_rate,
            end_learning_rate=learning_rate / 10,
            decay_steps=self.epochs * len(train_dataset),
        )

        # Optimizer
        optimizer = Adam(learning_rate=lr_schedule, epsilon=1e-8)

        # Checkpoint callback
        checkpoint_prefix = os.path.join(
            checkpoint_dir, "ckpt_{epoch}.weights.h5"
        )
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix, save_weights_only=True
        )

        # For a GPT-2 style language model
        model.compile(
            optimizer=optimizer,
            # This handles the logits properly for a language model
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            ),
            metrics=["accuracy"],
        )

        # Train model
        self.history = model.fit(
            train_dataset, epochs=self.epochs, callbacks=[checkpoint_callback]
        )

    # Function to generate text
    def generate_text(
        self,
        model,
        prompt,
        max_length=100,
        temperature=0.7,
    ):
        # Tokenize seed text
        input_ids = self.tokenizer.texts_to_sequences([prompt])[0]

        # Truncate or pad if necessary
        context_size = model.inputs[0].shape[1]
        if len(input_ids) > context_size:
            input_ids = input_ids[-context_size:]
        else:
            input_ids = [0] * (context_size - len(input_ids)) + input_ids

        generated_text = prompt
        input_ids = np.array([input_ids])

        # Generate text token by token
        for _ in range(max_length):
            predictions = model.predict(input_ids, verbose=0)[0]

            # Get the predictions for the last token
            predictions = predictions[-1] / temperature
            predicted_id = tf.random.categorical(
                tf.expand_dims(predictions, 0), num_samples=1
            )[-1, 0].numpy()

            # Update the input ids
            input_ids = np.roll(input_ids, -1, axis=1)
            input_ids[0, -1] = predicted_id

            # Convert token to word and add to generated text
            if predicted_id in self.index_word:
                word = self.index_word[predicted_id]
                generated_text += " " + word

                # Stop if we generate an end token
                if word == "<EOS>":
                    break

        return generated_text
