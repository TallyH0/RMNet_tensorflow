from model import RMNet_model
import importlib
from argparse import ArgumentParser
from os.path import splitext
from os.path import basename

if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--config', required=True, type=str)
  args = parser.parse_args()
  config = splitext(basename(args.config))[0]

  cfg = importlib.import_module(config)

  network = RMNet_model(cfg)
  network.build(train=True)
  network.train()
  