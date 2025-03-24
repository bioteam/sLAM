import argparse
from slam import slam_generator  # noqa: F401


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
    help="Name of model and JSON files",
)
parser.add_argument("-v", "--verbose", action="store_true", help="Verbose")
args = parser.parse_args()

generator = slam_generator(args.name)

result = generator.generate_text(args.prompt)
print(result)
