import tensorflow as tf
import numpy as np
import glob
import os
import random
import sys
import time
from tensorflow.keras import layers  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.optimizers.schedules import PolynomialDecay  # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint  # type: ignore  # noqa: F401


class slam_builder:
    """
    A simple language model based on transformer encoder/decoder architecture.
    Hyperparameters are set here before training (parameters are learned in the training,
    e.g. weights, values in the embedding matrices, coefficients in linear regression).
    """

    def __init__(
        self,
        quiet: bool = False,
        name: str = None,
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
        self.quiet = quiet
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
    def create_tokenizer(self):
        """Creates a simple WordPiece tokenizer."""
        if not self.quiet:
            print(
                "create_tokenizer(): makie a simple tokenizer (tf.keras.preprocessing.text.Tokenizer) with an estimated vocabulary size"
            )

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
            num_words=self.vocab_size,
            oov_token="<UNK>",  # Unknown token
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        )

    def fit(self, texts):
        """
        fit_on_texts():
        1. It scans through all the texts you provide
        2. It builds a word index dictionary (mapping words to integers
        3. It computes word frequencies and other metadata
        4. It doesn't return tokenized sequences - it just prepares the tokenizer
        """
        if not self.quiet:
            print(
                "fit(): create word-to-int and int-to-word indices and calculate word frequencies and other metadata"
            )
        self.tokenizer.fit_on_texts(texts)

        # After fitting, add special tokens to the word index properly
        word_index = self.tokenizer.word_index
        # Get the next available index
        next_index = len(word_index) + 1

        # Add special tokens
        self.tokenizer.word_index["<PAD>"] = 0  # Usually reserved for padding
        self.tokenizer.word_index["<BOS>"] = next_index
        self.tokenizer.index_word[next_index] = "<BOS>"
        next_index += 1
        self.tokenizer.word_index["<EOS>"] = next_index
        self.tokenizer.index_word[next_index] = "<EOS>"

        # Update vocabulary size
        self.vocab_size = next_index + 1
        if not self.quiet:
            print(f"fit(): actual vocabulary size is {self.vocab_size}")

    def load_text(self, input_dir, percentage):
        text = ""
        if not self.quiet:
            print("load_text(): read input text files and return 1 string")
        file_paths = glob.glob(f"{input_dir}/*")
        if percentage != 100:
            if percentage > 100:
                sys.exit("Invalid percentage: {percentage}")
            num_files = int(len(file_paths) * percentage / 100)
            file_paths = random.sample(file_paths, num_files)

        if not self.quiet:
            print(
                f"load_text(): using {percentage}% of files in {input_dir} for the dataset"
            )
        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8") as f:
                text += f.read() + "\n\n"

        return text

    def prepare_dataset(self, text):
        """
        Converts text to sequences of integers which are token
        ids that correspond to the indices in word_index.
        """
        if not self.quiet:
            print(
                "prepare_dataset(): tokenize, prepare input and target token sequences, and create a tf.data.Dataset.from_tensor_slices"
            )

        token_ids = self.tokenizer.texts_to_sequences([text])[0]
        self.num_tokens = len(token_ids)
        if not self.quiet:
            print(f"prepare_dataset(): number of tokens is {self.num_tokens}")
        """Create examples with context_size + 1 (inputs and targets)"""
        examples = []
        for i in range(0, self.num_tokens - self.context_size):
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
        """train_model

        Train the model

        Parameters
        ----------
        train_dataset : _type_
            _description_
        model : _type_
            _description_
        learning_rate : _type_, optional
            _description_, by default 5e-5
        checkpoint_dir : str, optional
            _description_, by default "./checkpoints"

        Total Parameters
        The total number of weights and biases in the entire model. This represents the complete set of values that define the model's behavior and what it has learned.

        Trainable Parameters
        These are parameters that are updated during the training process through backpropagation and gradient descent. They're the parts of the model that "learn" from the training data. Most parameters in a typical neural network are trainable.

        Non-Trainable Parameters
        These parameters are not updated during training. They remain fixed at their initial values or at values they were set to previously. Non-trainable parameters can come from:

        Layers that are explicitly frozen (set to non-trainable) during transfer learning
        Batch normalization statistics (moving means and variances) that are updated during training but not through backpropagation
        Embedding layers that are set to non-trainable (like when using pre-trained word embeddings)
        Parameters in layers where training is disabled

        An epoch represents one complete pass through the entire training dataset. Within each epoch, the training data is processed in smaller batches, with each batch being a "step."

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
        In the context of training GPT-2 style language models, "samples" has a specific meaning that differs from some other machine learning contexts:

        What Constitutes a Sample in LLM Training
        For GPT-2 style models:

        A sample is typically a sequence of tokens of a specific length (e.g., 512 or 1024 tokens)
        These sequences are often extracted from a larger corpus of text
        Each sample serves as a training example for the model to learn from
        Important Characteristics
        Sequence-based: Unlike image classification where one image = one sample, LLM samples are sequences of tokens

        Context window: The sample length is determined by the model's context window (maximum sequence length)

        Sliding windows: Samples may be created using sliding windows over text, potentially with overlap

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

        """For a GPT-2 style language model. This handles the logits properly for a LM"""
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

    def save(self, model):
        """Use a timestamp if there's no name"""
        if not self.name:
            self.name = time.strftime("%m-%d-%Y-%H-%M-%S", time.localtime())
        if not self.quiet:
            print(
                f"save(): saving Keras model ({self.name}.keras) and JSON file ({self.name}.json) with int-to-word decoding and metadata"
            )
        """In Tensorflow the tokenizer is not saved with the model, they must be saved separately"""
        model.save(f"{self.name}.keras")
        tokenizer_json = self.tokenizer.to_json()
        with open(f"{self.name}.json", "w", encoding="utf-8") as f:
            f.write(tokenizer_json)


class slam_generator:
    def __init__(self, name):
        self.name = name
        if not os.path.exists(f"{self.name}.json"):
            sys.exit(f"Tokenizer JSON file not found: {self.name}.json")
        with open(f"{self.name}.json", "r", encoding="utf-8") as f:
            tokenizer_json = f.read()
            self.tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(
                tokenizer_json
            )
        if not os.path.exists(f"{self.name}.keras"):
            sys.exit(f"Model file not found: {self.name}.keras")
        self.model = tf.keras.models.load_model(f"{self.name}.keras")

        self.index_word = {
            index: word for word, index in self.tokenizer.word_index.items()
        }
        # Add padding token if it exists in your tokenizer setup
        # self.index_word[0] = ""  # Often 0 is reserved for padding

        # Check tokenizer vocabulary size
        import json

        tokenizer_config = json.loads(tokenizer_json)
        print(
            f"Vocabulary size: {tokenizer_config.get('config', {}).get('num_words', len(tokenizer_config.get('word_index', {})))}"
        )
        print(
            f"Sample tokens: {list(tokenizer_config.get('word_index', {}).items())[:10]}"
        )

        # Verify the ID for <UNK> token
        print(
            f"<UNK> token id: {tokenizer_config.get('word_index', {}).get('<UNK>', 'Not found')}"
        )

    # 4. Function to convert a single ID to a word
    def id_to_word(self, token_id):
        return self.index_word.get(
            token_id, ""
        )  # Return empty string if ID not found

    # Function to generate text
    def generate_text(
        self,
        prompt,
        max_length=100,
        temperature=0.7,
    ):
        # embedding_layer = self.model.get_layer("token_embeddings")

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

        # self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
        #     num_words=embedding_layer.input_dim, oov_token="<UNK>"
        # )
        input_ids = self.tokenizer.texts_to_sequences([prompt])[0]

        # Truncate or pad if necessary
        context_size = self.model.inputs[0].shape[1]
        if len(input_ids) > context_size:
            input_ids = input_ids[-context_size:]
        else:
            input_ids = [0] * (context_size - len(input_ids)) + input_ids

        input_ids = np.array([input_ids])

        """Generate text token by token"""
        for _ in range(max_length):
            predictions = self.model.predict(input_ids, verbose=0)[0]

            # Get the predictions for the last token
            predictions = predictions[-1] / temperature
            predicted_id = tf.random.categorical(
                tf.expand_dims(predictions, 0), num_samples=1
            )[-1, 0].numpy()

            # Update the input ids
            input_ids = np.roll(input_ids, -1, axis=1)
            input_ids[0, -1] = predicted_id

            # Convert token to word and add to generated text
            # if predicted_id in self.index_word:
            word = self.id_to_word(predicted_id)
            prompt += " " + word

            # Stop if we generate an end token
            if word == "<EOS>":
                break

        return prompt


""" 
        
TensorFlow's SubwordTextEncoder is part of the tensorflow_text and tensorflow_datasets packages. It creates a subword vocabulary that can handle out-of-vocabulary words effectively. Here's how to use it:

import tensorflow as tf
import tensorflow_datasets as tfds

class SLAMWithSubwordTokenizer:
    def __init__(self, vocab_size=8000, quiet=False):
        self.vocab_size = vocab_size
        self.quiet = quiet
        self.tokenizer = None
        
    def fit(self, texts):
        if not self.quiet:
            print(f"Building subword tokenizer with target vocab size: {self.vocab_size}")
        
        # Create a subword tokenizer
        self.tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            corpus_iterator=texts,
            target_vocab_size=self.vocab_size,
            max_subword_length=20,  # Maximum length for subwords
            reserved_tokens=["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
        )
        
        # The actual vocabulary size might differ from requested
        self.vocab_size = self.tokenizer.vocab_size
        
        if not self.quiet:
            print(f"Actual vocabulary size: {self.vocab_size}")
            print(f"Sample tokens: {self.tokenizer.subwords[:10]}")
        
        return self
    
    def encode(self, texts, add_special_tokens=True):
        if not isinstance(texts, list):
            texts = [texts]
            
        encoded = []
        for text in texts:
            # Encode the text to subword IDs
            tokens = self.tokenizer.encode(text)
            
            # Add special tokens if requested
            if add_special_tokens:
                bos_id = self.tokenizer.vocab_size - 3  # <BOS> index
                eos_id = self.tokenizer.vocab_size - 2  # <EOS> index
                tokens = [bos_id] + tokens + [eos_id]
                
            encoded.append(tokens)
            
        return encoded
    
    def decode(self, sequences):
        if not isinstance(sequences, list):
            sequences = [sequences]
            
        decoded = []
        for seq in sequences:
            # Remove special tokens if present (simple approach)
            tokens = [t for t in seq if t < self.tokenizer.vocab_size - 4]  # Exclude special tokens
            
            # Decode back to text
            text = self.tokenizer.decode(tokens)
            decoded.append(text)
            
        return decoded if len(decoded) > 1 else decoded[0]
    
    def save(self, filename):
        self.tokenizer.save_to_file(filename)
        if not self.quiet:
            print(f"Tokenizer saved to {filename}")
    
    def load(self, filename):
        self.tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(filename)
        self.vocab_size = self.tokenizer.vocab_size
        if not self.quiet:
            print(f"Loaded tokenizer with vocab size: {self.vocab_size}")
        return self


# Example usage
if __name__ == "__main__":
    # Sample texts
    texts = [
        "This is a sample sentence.",
        "Another example with different words.",
        "Subword tokenization handles unknown words better.",
        "Words like 'tokenization' are split into smaller parts."
    ]
    
    # Create and fit the tokenizer
    tokenizer = SLAMWithSubwordTokenizer(vocab_size=100)
    tokenizer.fit(texts)
    
    # Encode some text
    encoded = tokenizer.encode("This is a new example.")
    print(f"Encoded: {encoded}")
    
    # Decode back to text
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")
    
    # Save and load
    tokenizer.save("my_subword_tokenizer")
    new_tokenizer = SLAMWithSubwordTokenizer()
    new_tokenizer.load("my_subword_tokenizer")


"""
