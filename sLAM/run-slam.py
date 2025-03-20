import glob
import argparse
import time
from slam import slam_builder
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--input_dir",
    required=True,
    help="Directory with text files",
)
parser.add_argument("-v", "--verbose", action="store_true", help="Verbose")
args = parser.parse_args()


def main():
    # Get your text files
    file_paths = glob.glob(f"{args.input_dir}/*.md")

    builder = slam_builder()

    # Load data
    all_texts = []
    for file_path in file_paths[:100]:  # Limit to 100 files for this example
        with open(file_path, "r", encoding="utf-8") as f:
            all_texts.append(f.read())

    # Create tokenizer
    word_index, index_word = builder.tokenize(all_texts)

    token_ids = builder.tokenizer.texts_to_sequences([all_texts])[0]

    # Adjust vocab_size to actual vocabulary size
    actual_vocab_size = len(word_index) + 1
    print(f"Actual vocabulary size: {actual_vocab_size}")

    # Prepare dataset
    # dataset = builder.prepare_dataset(token_ids)

    # Convert token_ids to a TensorFlow tensor
    token_ids_tensor = tf.convert_to_tensor(token_ids, dtype=tf.int32)

    # Create sequences and targets for training
    sequence_length = 50  # Adjust as needed
    dataset = tf.data.Dataset.from_tensor_slices(token_ids_tensor)

    # Create sequences and target pairs
    dataset = dataset.window(sequence_length + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(
        lambda window: window.batch(sequence_length + 1)
    )

    # Split into inputs and targets
    dataset = dataset.map(lambda x: (x[:-1], x[1:]))

    # Batch the dataset
    batch_size = 64  # Adjust as needed
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Create model
    model = builder.create_small_gpt2_model()

    # Print model summary
    model.summary()

    # Train model
    builder.train_model(model, dataset)

    # Generate sample text
    sample_text = builder.generate_text(
        model, index_word, "Once upon a time", max_length=50
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
    file_paths = glob.glob("your_data_directory/*.txt")



    # Load data
    all_texts = []
    for file_path in file_paths[:100]:  # Limit to 100 files for this example
        with open(file_path, "r", encoding="utf-8") as f:
            all_texts.append(f.read())

    # Create tokenizer
    word_index, index_word = create_simple_tokenizer(
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


if __name__ == "__main__":
    main()

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
