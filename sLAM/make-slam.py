import argparse
from slam import slam_builder


parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--input_dir",
    required=True,
    help="Directory with text files",
)
parser.add_argument(
    "-p",
    "--percent_files",
    default=100,
    type=int,
    help="Percentage of files used in dataset",
)
parser.add_argument(
    "-n",
    "--name",
    help="Default is timestamp of completion",
)
parser.add_argument("-q", "--quiet", action="store_true", help="Not verbose")
args = parser.parse_args()


builder = slam_builder(quiet=args.quiet, name=args.name)

texts = builder.load_text(args.input_dir, args.percent_files)

builder.create_tokenizer()

builder.fit(texts)

dataset = builder.prepare_dataset(texts)

model = builder.create_small_gpt2_model()

builder.train_model(dataset, model)

if not args.quiet:
    model.summary()
    embedding_layer = model.get_layer("token_embeddings")
    print(f"Vocabulary size: {embedding_layer.input_dim}")

builder.save(model)
