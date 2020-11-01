from model import RMNet_model
import importlib
from argparse import ArgumentParser
from os.path import splitext
from os.path import basename
from os.path import join
import cv2
import numpy as np

def GetDistance(tensor0, tensor1):
  dot_product = np.sum(np.multiply(tensor0, tensor1))
  norm0 = np.sqrt(np.sum(np.square(tensor0)))
  norm1 = np.sqrt(np.sum(np.square(tensor1)))

  return dot_product / (norm0 * norm1)

if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--config', required=True, type=str)
  parser.add_argument('--data_dir', required=True, type=str)
  parser.add_argument('--txt_query', required=True, type=str)
  parser.add_argument('--txt_test', required=True, type=str)
  args = parser.parse_args()
  config = splitext(basename(args.config))[0]

  cfg = importlib.import_module(config)

  network = RMNet_model(cfg)
  network.build(train=False)
  network.load()

  discriptor_query = []
  discriptor_test = []

  with open(args.txt_query) as f:
    for line in f:
      parse = line.strip().split('\t')
      discriptor = [parse[0], int(parse[1]), 0]
      discriptor_query.append(discriptor)

  with open(args.txt_test) as f:
    for line in f:
      parse = line.strip().split('\t')
      discriptor = [parse[0], int(parse[1]), 0]
      discriptor_test.append(discriptor)

  for query in discriptor_query:
    img = cv2.imread(join(args.data_dir, query[0]))
    print(query[0])
    emb = network.inference(img)
    query[2] = emb

  for test in discriptor_test:
    img = cv2.imread(join(args.data_dir, test[0]))
    print(test[0])
    emb = network.inference(img)
    test[2] = emb

  acc_rank1 = 0
  for query in discriptor_query:
    distances = []
    for test in discriptor_test:
      print(query[0], test[0])
      distances.append(GetDistance(query[2], test[2]))

    index = np.argmax(distances)
    if query[1] == discriptor_test[index][1]:
      acc_rank1 += 1

  
  print('Rank-1 accuracy : ', acc_rank1 / len(discriptor_query))

  