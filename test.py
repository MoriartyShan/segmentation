import copy
import os
import argparse
import torch
import time
import random
import cv2
import models.model as mm

import dataset
from loss import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='',
      help='path to images to be detected')
    parser.add_argument('--seed', type=int, default=-1,
      help='to random seed, if not set, use time to generate one')
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    parser.add_argument('--device', type=str, default='cpu',
                        help='device use to train, cpu or cuda:0')
    args = parser.parse_args()
    print("current args is %s" %(str(args)))
    return args

def createTrainPath():
  if not (os.path.exists("./run")):
    os.mkdir("./run")
  prefix = "./run/test_"
  id = 0
  while(True):
    path = prefix + str(id)
    if (os.path.exists(path)):
      id += 1
      continue
    else:
      os.mkdir(path)
      return path


def main():
  runtime_path = createTrainPath()

  args = parse_args()
  default_type = torch.float32
  torch.set_default_dtype(default_type)

  root = args.data
  max_size = (2048, 1024)#dataset._image_size #height, width

  device = torch.device(args.device)
  seed = args.seed
  if (seed < 0):
    seed = int(time.time())
  print("random seed %d" % seed)
  random.seed(seed)
  torch.manual_seed(seed)

  images = os.listdir(root)

  if len(args.weights) == 0:
    print("--weights is not input")
    exit(0)
  else:
    print("loading weight from %s" %args.weights)
    model = mm.TorchNet()
    # model.load_state_dict(torch.load('test.pt', map_location = device))
    model = torch.load(args.weights, map_location = device)

  model = model.to(device)

  model.eval()
  for name in images:
    image = cv2.imread(os.path.join(root, name), cv2.IMREAD_COLOR)

    new_size = dataset.Sample.get_new_size(image.shape[0:2], max_size)
    image = cv2.resize(image, (new_size[1], new_size[0]))

    preprocessed = dataset.Sample.preprocess_image(image)
    preprocessed = torch.from_numpy(preprocessed).to(device).unsqueeze(dim=0)
    output = model(preprocessed).squeeze(dim=0).squeeze(dim=0)
    # output = torch.sigmoid(model(preprocessed)['out']).squeeze(dim=0).squeeze(dim=0)
    print("shape %s" %str(output.shape))

    prefix = name.split('.')[0]

    for i in range(3):
      threshold = 0.5 + i * 0.2
      overlaped = dataset.Sample.draw_segmentation(copy.deepcopy(image), output, threshold)
      cv2.imwrite(os.path.join(runtime_path, prefix + '_' + str(threshold) + '.png'), overlaped)


main()
