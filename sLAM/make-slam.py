import argparse
from slam import slam_builder
from datasets import load_dataset


parser = argparse.ArgumentParser()
parser.add_argument(
    "-t",
    "--text_percentage",
    default=1,
    type=int,
    help="Percentage of wikitext-2-v1 used to make dataset",
)
parser.add_argument(
    "--context_size",
    default=32,
    type=int,
    help="Context size",
)
parser.add_argument(
    "-n",
    "--name",
    help="Name used to save files, default is timestamp of completion",
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
parser.add_argument(
    "--d_model",
    type=int,
    default=256,
    help="Number of embedding dimensions",
)
parser.add_argument(
    "-d",
    "--download",
    choices=["wikitext-2-v1", "cc_news"],
    help="Dataset to download",
    default="cc_news",
)
parser.add_argument(
    "--num_rows",
    type=int,
    help="Number of rows to download from cc_news",
    default=5000,
)
parser.add_argument(
    "--use_mlflow",
    help="Use MLFlow",
    action="store_true",
)
parser.add_argument("-p", "--prompt", help="Prompt", required=True)
parser.add_argument("-v", "--verbose", action="store_true", help="Verbose")
args = parser.parse_args()


builder = slam_builder(
    verbose=args.verbose,
    name=args.name,
    epochs=args.epochs,
    context_size=args.context_size,
    d_model=args.d_model,
    temperature=args.temperature,
    use_mlflow=args.use_mlflow,
    download=args.download,
    num_rows=args.num_rows,
)
if args.download == "wikitext-2-v1":
    wp_texts = load_dataset("wikitext", "wikitext-2-v1")
    texts = builder.clean_wikitext(wp_texts, args.text_percentage)
elif args.download == "cc_news":
    cc_texts = load_dataset("cc_news", split=f"train[:{args.num_rows}]")
    texts = builder.clean_cc_news(cc_texts)

if args.verbose:
    builder.analyze_text(texts)

builder.create_tokenizer()

builder.adapt(texts)

if args.use_mlflow:
    builder.start_mlflow_server()

train_dataset, val_dataset = builder.prepare_datasets(texts)

model = builder.create_small_gpt2_model()

model = builder.train_model(train_dataset, val_dataset, model)

builder.save(model)

if args.verbose:
    model.summary()

result = builder.generate_text(args.prompt, model)
print(f"Result: {result}")

if args.use_mlflow:
    builder.stop_mlflow_server()
