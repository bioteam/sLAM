import glob


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
