import argparse
import sys
from slam import slam_builder
from datasets import load_dataset


parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--input_dir",
    help="Specify a directory with text files",
)
parser.add_argument(
    "-d",
    "--download",
    action="store_true",
    help="Do a text download from Hugging Face",
)
parser.add_argument(
    "-t",
    "--text_percentage",
    default=100,
    type=int,
    help="Percentage of input text used to make dataset",
)
parser.add_argument(
    "-n",
    "--name",
    help="Name used to save files, default is timestamp of completion",
)
parser.add_argument(
    "--min_sentence_len",
    type=int,
    default=32,
    help="Mininum sentence length used in training",
)
parser.add_argument(
    "--temperature",
    type=float,
    default=0.7,
    help="Temperature used for generation",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=3,
    help="Number of epochs",
)
parser.add_argument("-p", "--prompt", help="Prompt", required=True)
parser.add_argument("-v", "--verbose", action="store_true", help="Verbose")
args = parser.parse_args()


builder = slam_builder(
    verbose=args.verbose, name=args.name, epochs=args.epochs
)

if args.download:
    raw_texts = load_dataset("wikitext", "wikitext-2-v1")
    texts = builder.clean_wikitext(
        raw_texts, args.text_percentage, args.min_sentence_len
    )
elif args.input_dir:
    texts = builder.load_text(args.input_dir, args.text_percentage)
else:
    sys.exit("No input")

builder.create_tokenizer()

builder.adapt(texts)

if args.verbose:
    builder.analyze_text(texts)

# train_dataset, val_dataset = builder.prepare_datasets(texts)
train_dataset = builder.prepare_datasets(texts)

model = builder.create_small_gpt2_model()

# builder.train_model(train_dataset, val_dataset, model)
builder.train_model(train_dataset, model)

if args.verbose:
    model.summary()
    embedding_layer = model.get_layer("token_embeddings")
    print(f"Vocabulary size: {embedding_layer.input_dim}")
    print(f"Number of tokens: {builder.num_tokens}")

result = builder.generate_text(
    model, args.prompt, temperature=args.temperature
)
print(f"Result: {result}")

builder.save(model)
