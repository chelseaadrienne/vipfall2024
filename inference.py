from argparse import ArgumentParser
from minimagen.generate import load_minimagen, sample_and_save

parser = ArgumentParser()
parser.add_argument("-d", "--TRAINING_DIRECTORY", dest = "TRAINING_DIRECTORY", help = "Training directory to use for inference", type = str)
args = parser.parse_args()

captions = ['a 90s movie poster']
sample_and_save(captions, training_directory = args.TRAINING_DIRECTORY)

minimagen = load_minimagen(args.TRAINING_DIRECTORY)
sample_and_save(captions, minimagen = minimagen)
