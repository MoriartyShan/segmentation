import torch
from models.model import *
import argparse


def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument('--input', type=str, default='',
                    help='input weights path')
  parser.add_argument('--output', type=str, default='',
                    help='output module path')
  parser.add_argument('--device', type=str, default='cpu',
                    help='device use to trace, cpu or cuda:0')
  args = parser.parse_args()
  print("current args is %s" %(str(args)))
  if len(args.input) == 0:
    print("--input is not assigned")
    exit(0)
  if len(args.output) == 0:
    print("--output is not assigned")
    exit(0)

    return args
def main():
  args = parse_args()
  model = torch.load(args.input, map_location="cpu")
  example = torch.rand(1, 3, 512, 512)

  model.eval()
  ScriptModule = torch.jit.trace(model, example)
  torch.jit.save(ScriptModule, args.output)

main()
