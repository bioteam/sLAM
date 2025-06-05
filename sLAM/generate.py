import argparse
from slam import slam_builder  # noqa: F401


parser = argparse.ArgumentParser()
parser.add_argument(
    "-p",
    "--prompt",
    required=True,
    help="Text prompt",
)
parser.add_argument(
    "-n",
    "--name",
    required=True,
    help="Name of model and tokenizer *pickle files",
)
parser.add_argument(
    "--temperature",
    type=float,
    default=0.7,
    help="Temperature used for generation",
)
parser.add_argument("-v", "--verbose", action="store_true", help="Verbose")
args = parser.parse_args()

generator = slam_builder(verbose=args.verbose, temperature=args.temperature)

model = generator.load(args.name)

result = generator.generate_text(args.prompt, model)
print(result)
