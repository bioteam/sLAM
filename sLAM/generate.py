import argparse
import pickle
from slam import slam_builder  # noqa: F401


parser = argparse.ArgumentParser()
parser.add_argument(
    "-p",
    "--prompt",
    required=True,
    help="Text prompt",
)
parser.add_argument(
    "-m",
    "--model",
    required=True,
    help="Model file",
)
parser.add_argument(
    "-s",
    "--slam_builder_file",
    required=True,
    help="Pickle file",
)
parser.add_argument("-v", "--verbose", action="store_true", help="Verbose")
args = parser.parse_args()

try:
    with open(args.slam_builder_file, "rb") as f:
        builder = pickle.load(f)
except Exception as e:
    print(f"Error reading pickle file: {e}")

text = builder.generate_text(args.model, args.prompt)
print(text)
