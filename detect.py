import torch
import cv2
from dataset import Sample
import copy
import os
import time
import argparse

max_size = (1024, 1024)#dataset._image_size #height, width

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='',
                        help='input jit model path')
    parser.add_argument('--image', type=str, default='',
                        help='image path')
    parser.add_argument('--output', type=str, default='',
                        help='output detected path')
    parser.add_argument('--device', type=str, default='cpu',
                        help='device use to detect, cpu or cuda:0')
    args = parser.parse_args()
    print("current args is %s" %(str(args)))
    if len(args.model) == 0:
        print("--input is not assigned")
        exit(0)
    if len(args.image) == 0:
        print("--output is not assigned")
        exit(0)

    return args
def main():
  args = parse_args()
  round = 100

  prefix = "hello"
  device = torch.device(args.device)
  model = torch.jit.load(args.model, map_location=device)
  model.eval()
  with torch.no_grad():
    for r in range(round):
      begin = time.time()
      image = cv2.imread(args.image)
      new_size = Sample.get_new_size(image.shape[0:2], max_size)
      image = cv2.resize(image, (new_size[1], new_size[0]))
      preprocessed = Sample.preprocess_image(image)
      preprocessed = torch.from_numpy(preprocessed).to(device).unsqueeze(dim=0)
      output = model(preprocessed).squeeze(dim=0).squeeze(dim=0)
      # output = torch.sigmoid(model(preprocessed)['out']).squeeze(dim=0).squeeze(dim=0)
      # print("shape %s" % str(output.shape))
      if (r == 0):
        for i in range(3):
          threshold = 0.5 + i * 0.2
          overlaped = Sample.draw_segmentation(copy.deepcopy(image), output, threshold)
          cv2.imwrite(os.path.join(args.output, prefix + '_' + str(threshold) + '.png'), overlaped)
      print("cost time ", time.time() - begin, "sec")

main()
