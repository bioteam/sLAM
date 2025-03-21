import argparse
import glob
import pickle
import time
from slam import slam_builder


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


parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--input_dir",
    required=True,
    help="Directory with text files",
)
parser.add_argument("-v", "--verbose", action="store_true", help="Verbose")
args = parser.parse_args()


builder = slam_builder(verbose=args.verbose)

# Get your text files
file_paths = glob.glob(f"{args.input_dir}/*.md")

all_texts = builder.load_text(file_paths)

builder.create_simple_tokenizer()

builder.fit(all_texts)

# Tokenize and prepare dataset
some_texts = builder.load_text(file_paths[:100])
dataset = builder.prepare_dataset(some_texts)

# Create model
model = builder.create_small_gpt2_model()

# Print model summary
model.summary()

# Train model
builder.train_model(dataset, model)

timestamp = time.strftime("%m-%d-%Y-%H-%M-%S", time.localtime())
model.save(f"{args.input_dir}/{timestamp}.keras")
with open(f"{args.input_dir}/builder-{timestamp}.bin", "wb") as f:
    pickle.dump(slam_builder, f)


"""
    # Generate sample text
    sample_text = generate_text(
        model, tokenizer, index_word, "Once upon a time", max_length=50
    )
    print("Generated text:")
    print(sample_text)

"""
