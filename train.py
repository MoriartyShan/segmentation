import time
import torch

from pathlib import Path
import sys
import argparse

from dataset import *
from models.model import *
import loss


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative



def train(model:torch.nn.Module, data_loader:torch.utils.data.DataLoader, optimizer,
          compute_loss, epoch, device):
  size = len(data_loader)

  print("num of size %d" %(size))

  show_count = (size - 1) // 10
  show_count = 1 if (show_count == 0) else show_count
  show_count = 100 if (show_count > 100) else show_count
  total_loss = 0.0
  total_iou = 0.0
  model.train()
  epoch_begin = time.time()
  for index, batch_data in enumerate(data_loader):
    image = batch_data[0].to(device)
    label = batch_data[1].to(device)

    optimizer.zero_grad()
    output = model(image)
    '''
    print("output type %s" %(str(type(output))))
    print("output type %s" % (str(type(output['out']))))
    print("output: shape %s, type %s" %(str(output['out'].shape), str(output['out'].dtype)))
    print("label: shape %s, type %s" % (str(label.shape), str(label.dtype)))
    '''
    # print("shapes ", output.dtype, label.dtype)
    bce, iou = compute_loss(output, label)

    total_iou += iou.data
    total_loss += bce.data

    bce.backward()
    optimizer.step()
    if ((index % show_count) == 0):
      num = index + 1
      cost = time.time() - epoch_begin
      print('train %d in ep %d %0.2f%%, mean loss %f, current loss %f, mean iou %f, current iou %f, cost %f sec, left %f sec' % (
        index, epoch, 100.0 * num / size, total_loss / num, bce.data, total_iou / num, iou.data,
        cost, cost / num * (size - num)))
  cost = time.time() - epoch_begin
  print('epoch %d finished, trained %d samples, mean loss %f, mean iou %f, cost time %f sec' % (
    epoch, size, total_loss / size, total_iou / size, cost))
  return total_loss / size, total_iou / size, cost



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=-1, help='to random seed, if not set, use time to generate one')
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--lr', type=float, default=0.00001, help='beginning of learning rate')
    parser.add_argument('--device', type=str, default='cpu',
                        help='device use to train, cpu or cuda:0')
    args = parser.parse_args()
    print("current args is %s" %(str(args)))
    return args

def main():
  args = parse_args()
  default_type = torch.float32

  dataset_path = '/home/moriarty/Projects/x.csv'
  image_path = '/home/moriarty/Datasets/coco/train2017'
  dataset = loadDataset(dataset_path)
  dataset = filter(dataset)
  dataset = Dataset(dataset, image_path)
  size = len(dataset)
  train_size = int(0.9 * size)

  batch_size = args.batch_size
  lr = args.lr
  device = torch.device(args.device)

  model = TorchNet()
  # model = Model()
  model = model.to(device)
  compute_loss = loss.MultipleLoss()

  trset, teset = torch.utils.data.random_split(
    dataset,
    [train_size, size - train_size],
    generator=torch.Generator().manual_seed(42))

  trloader = torch.utils.data.DataLoader(
    trset,
    batch_size=batch_size,
    shuffle=True,
    sampler=None,
    batch_sampler=None,
    num_workers=4,
    collate_fn=None,
    pin_memory=True,
    drop_last=True,
    timeout=0,
    worker_init_fn=None,
    prefetch_factor=4,
    persistent_workers=True)

  teloader = torch.utils.data.DataLoader(
    teset,
    batch_size=batch_size,
    shuffle=True,
    sampler=None,
    batch_sampler=None,
    num_workers=4,
    collate_fn=None,
    pin_memory=True,
    drop_last=True,
    timeout=0,
    worker_init_fn=None,
    prefetch_factor=4,
    persistent_workers=True)

  optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                               betas=(0.9, 0.99))

  for i in range(20):
    train(model, trloader, optimizer, compute_loss, i, device)
  torch.save(model, 'model.pt')


main()




