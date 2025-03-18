import argparse
import tensorflow as tf
import numpy as np
import os
import glob
import time
from tensorflow.keras import layers  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.optimizers.schedules import PolynomialDecay  # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint  # type: ignore  # noqa: F401

# import tensorflow_text as tf_text  # type: ignore

# Set memory growth to avoid OOM issues
physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--input_dir",
    required=True,
    help="Directory with text files",
)
parser.add_argument("-v", "--verbose", action="store_true", help="Verbose")
args = parser.parse_args()


# Define the transformer block
def transformer_block(x, n_heads, d_model, d_ff, dropout_rate):
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
    # Hyperparameters are set here before training (parameters are learned in the training,
    # e.g. weights, values in the embedding matrices, coefficients in linear regression).
    vocab_size=10000,
    context_size=256,
    d_model=256,
    n_layers=4,
    n_heads=4,
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
        shape=(context_size,), dtype=tf.int32, name="input_ids"
    )

    # Embedding layer
    token_embeddings = layers.Embedding(
        input_dim=vocab_size, output_dim=d_model, name="token_embeddings"
    )(input_ids)

    # Positional embeddings
    positions = tf.range(start=0, limit=context_size, delta=1)
    position_embeddings = layers.Embedding(
        input_dim=context_size, output_dim=d_model, name="position_embeddings"
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
    for i in range(n_layers):
        x = transformer_block(x, n_heads, d_model, d_ff, dropout_rate)

    # Output projection
    logits = layers.Dense(vocab_size, name="logits")(x)

    # Create model
    model = tf.keras.Model(inputs=input_ids, outputs=logits)

    return model


# Create a simple tokenizer
def create_simple_tokenizer(texts, vocab_size=10000):
    """Creates a simple WordPiece tokenizer."""
    vocab_size = min(vocab_size, 30000)  # Limit vocab for efficiency

    # bert_tokenizer = tf_text.BertTokenizer(
    #     vocab_lookup_table=None,
    #     suffix_indicator="##",
    #     max_bytes_per_word=100,
    #     max_chars_per_token=None,
    #     token_out_type=tf.int64,
    #     unknown_token="[UNK]",
    #     split_unknown_characters=False,
    # )

    # If you prefer to use a simpler tokenizer:
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=vocab_size, oov_token="<UNK>"
    )
    tokenizer.fit_on_texts(texts)

    # Add special tokens
    word_index = tokenizer.word_index
    word_index["<PAD>"] = 0  # Padding token
    word_index["<BOS>"] = len(word_index) + 1  # Beginning of sequence
    word_index["<EOS>"] = len(word_index) + 1  # End of sequence

    # Reverse the word index for decoding
    index_word = {v: k for k, v in word_index.items()}

    return tokenizer, word_index, index_word


# Function to prepare the dataset
def prepare_dataset(file_paths, tokenizer, context_size=256, batch_size=4):
    # Load and concatenate all text
    all_text = ""
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            all_text += f.read() + "\n\n"

    # Tokenize text
    token_ids = tokenizer.texts_to_sequences([all_text])[0]

    # Create examples with context_size + 1 (inputs and targets)
    examples = []
    for i in range(0, len(token_ids) - context_size):
        examples.append(token_ids[i : i + context_size + 1])
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
    dataset = dataset.shuffle(10000).batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


# Custom loss function to handle logits
def loss_function(target_ids, logits):
    # Using sparse categorical crossentropy for efficiency
    # We have 3D logits [batch, sequence, vocab] and 2D targets [batch, sequence]
    mask = tf.math.not_equal(target_ids, 0)  # Mask padding tokens
    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )(target_ids, logits)

    # Apply mask and calculate mean
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


# Custom training function with callbacks
def train_model(
    model,
    train_dataset,
    epochs=3,
    learning_rate=5e-5,
    checkpoint_dir="./checkpoints",
):
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Learning rate schedule
    lr_schedule = PolynomialDecay(
        initial_learning_rate=learning_rate,
        end_learning_rate=learning_rate / 10,
        decay_steps=epochs * len(train_dataset),
    )

    # Optimizer
    optimizer = Adam(learning_rate=lr_schedule, epsilon=1e-8)

    # Checkpoint callback
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}.weights.h5")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix, save_weights_only=True
    )

    # Compile model
    model.compile(
        optimizer=optimizer, loss=loss_function, metrics=["accuracy"]
    )

    # Train model
    history = model.fit(
        train_dataset, epochs=epochs, callbacks=[checkpoint_callback]
    )

    return history


# Function to generate text
def generate_text(
    model, tokenizer, index_word, seed_text, max_length=100, temperature=0.7
):
    # Tokenize seed text
    input_ids = tokenizer.texts_to_sequences([seed_text])[0]

    # Truncate or pad if necessary
    context_size = model.inputs[0].shape[1]
    if len(input_ids) > context_size:
        input_ids = input_ids[-context_size:]
    else:
        input_ids = [0] * (context_size - len(input_ids)) + input_ids

    generated_text = seed_text
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
        if predicted_id in index_word:
            word = index_word[predicted_id]
            generated_text += " " + word

            # Stop if we generate an end token
            if word == "<EOS>":
                break

    return generated_text


def main():
    # Parameters
    vocab_size = 10000
    context_size = 256
    d_model = 256
    n_layers = 4
    n_heads = 4
    batch_size = 4
    epochs = 3

    # Get your text files
    file_paths = glob.glob(f"{args.input_dir}/*.md")

    # Load data
    all_texts = []
    for file_path in file_paths[:100]:  # Limit to 100 files for this example
        with open(file_path, "r", encoding="utf-8") as f:
            all_texts.append(f.read())

    # Create tokenizer
    tokenizer, word_index, index_word = create_simple_tokenizer(
        all_texts, vocab_size
    )

    # Adjust vocab_size to actual vocabulary size
    actual_vocab_size = len(word_index) + 1
    print(f"Actual vocabulary size: {actual_vocab_size}")

    # Prepare dataset
    dataset = prepare_dataset(
        file_paths[:100], tokenizer, context_size, batch_size
    )

    # Create model
    model = create_small_gpt2_model(
        vocab_size=actual_vocab_size,
        context_size=context_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
    )

    # Print model summary
    model.summary()

    # Train model
    train_model(model, dataset, epochs=epochs)

    # Generate sample text
    sample_text = generate_text(
        model, tokenizer, index_word, "Once upon a time", max_length=50
    )
    print("Generated text:")
    print(sample_text)
    """
    Generated text:
    Once upon a time if this mode mode seconds default is <UNK> mode bring the fastest but 
    if the traffic dns address name parameter must be used for this parameter is 0 0 0 no number 
    of traffic mode by traffic url no matter how many traffic bytes used for each mode if it
    """
    model.save(
        f"{args.input_dir}/{time.strftime('%m-%d-%Y-%H-%M-%S', time.localtime())}.keras"
    )


if __name__ == "__main__":
    main()

"""
# Even smaller model for extremely limited resources
model = create_small_gpt2_model(
    vocab_size=5000,       # Smaller vocabulary
    context_size=128,      # Shorter context
    d_model=128,           # Smaller embeddings
    n_layers=2,            # Fewer layers
    n_heads=2,             # Fewer attention heads
    d_ff=512               # Smaller feed-forward
)

Mixed precision training
# Add at the beginning of your script
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
"""
