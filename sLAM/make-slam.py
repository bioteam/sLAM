import argparse
import glob
import pickle
import time
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
    "--percent_texts",
    default=100,
    type=int,
    help="Percentage of text used in dataset",
)
parser.add_argument("-v", "--verbose", action="store_true", help="Verbose")
args = parser.parse_args()


builder = slam_builder(verbose=args.verbose)

file_paths = glob.glob(f"{args.input_dir}/*.md")

all_texts = builder.load_text(file_paths)

builder.create_simple_tokenizer()

builder.fit(all_texts)

dataset = builder.prepare_dataset(all_texts, args.percent_texts)

model = builder.create_small_gpt2_model()

model.summary()

builder.train_model(dataset, model)

timestamp = time.strftime("%m-%d-%Y-%H-%M-%S", time.localtime())
model.save(f"slam-{timestamp}.keras")
with open(f"builder-{timestamp}.bin", "wb") as f:
    pickle.dump(builder, f)
