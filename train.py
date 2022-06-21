import time
import torch

from pathlib import Path
import sys
import argparse
import random

from dataset import *
from models.model import *
import loss as Loss


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def createTrainPath():
  if not (os.path.exists("./run")):
    os.mkdir("./run")
  prefix = "./run/train_"
  id = 0
  while(True):
    path = prefix + str(id)
    if (os.path.exists(path)):
      id += 1
      continue
    else:
      os.mkdir(path)
      return path


def test(model:torch.nn.Module, data_loader:torch.utils.data.DataLoader,
          compute_loss, epoch, device):
  with torch.no_grad():
    size = len(data_loader)

    print("num of test size %d" %(size))

    show_count = (size - 1) // 10
    show_count = 1 if (show_count == 0) else show_count
    show_count = 100 if (show_count > 100) else show_count
    total_loss = 0.0
    total_iou = 0.0
    model.eval()
    epoch_begin = time.time()
    for index, batch_data in enumerate(data_loader):
      image = batch_data[0].to(device)
      label = batch_data[1].to(device)

      output = model(image)
      bce, iou = compute_loss(output, label)

      total_iou += iou.data
      total_loss += bce.data

      if ((index % show_count) == 0):
        num = index + 1
        cost = time.time() - epoch_begin
        print('test %d in ep %d %0.2f%%, mean loss %.04f, current loss %.04f, '
              'mean iou %.04f, current iou %.04f, cost %.02f sec, left %.02f sec' % (
          index, epoch, 100.0 * num / size, total_loss / num, bce.data,
          total_iou[-1] / num, iou.data[-1],
          cost, cost / num * (size - num)))
    cost = time.time() - epoch_begin
    print('test epoch %d finished, test %d samples, mean loss %.04f,'
          ' mean iou %.04f, cost time %.02f sec' % (
      epoch, size, total_loss / size, total_iou[-1] / size, cost))
    return total_loss / size, total_iou / size, cost

def train(model:torch.nn.Module, data_loader:torch.utils.data.DataLoader, optimizer,
          compute_loss, epoch, device):
  size = len(data_loader)

  print("num of train size %d" %(size))

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
    # print("image shape", image.shape, torch.max(image), torch.min(image))

    optimizer.zero_grad()
    output = model(image)
    '''
    print("output type %s" % (str(type(output))))
    print("output: shape %s, type %s" %(str(output.shape), str(output.dtype)))
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
      print('train %d in ep %d %0.2f%%, mean loss %.04f, current loss %.04f, '
            'mean iou %.04f, current iou %.04f, cost %.02f sec, left %.02f sec' % (
        index, epoch, 100.0 * num / size, total_loss / num, bce.data,
        total_iou[-1] / num, iou.data[-1],
        cost, cost / num * (size - num)))
  cost = time.time() - epoch_begin
  print('train epoch %d finished, trained %d samples, mean loss %.04f, '
        'mean iou %.04f, cost time %.02f sec' % (
    epoch, size, total_loss / size, total_iou[-1] / size, cost))
  return total_loss / size, total_iou / size, cost



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=-1,
      help='to random seed, if not set, use time to generate one')
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    parser.add_argument('--cfg', type=str, default='',
                        help='model.yaml path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16,
                        help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--device', type=str, default='cpu',
                        help='device use to train, cpu or cuda:0')
    args = parser.parse_args()
    print("current args is %s" %(str(args)))
    return args

class LossManager():
  def __init__(self, is_train:bool, iou_num = -1):
    self.is_train = is_train
    self.iou_num = iou_num

    self.best_iou = None
    self.best_loss = None
    self.best_epoch = None

    self.iou = None
    self.loss = None
    self.epoch = -1

  def update(self, loss, iou, epoch):
    loss = loss.detach()
    iou = iou.detach()

    self.iou = iou
    self.loss = loss

    if (self.epoch < 0):
      self.epoch = epoch
      self.best_loss = loss
      self.best_iou = iou
      self.best_epoch = epoch
      return False

    self.epoch = epoch
    if (loss < self.best_loss):
      self.best_iou = iou
      self.best_epoch = epoch
      self.best_loss = loss
      return True
    return False

  def init_file(self, file):
    #epoch,best_train_epoch,current_train_loss,best_train_loss,current_and_best_train_iou,
    #best_test_epoch,current_test_loss, best_test_loss, current_and_best_test_iou,

    strain = 'epoch,best_train_epoch,current_train_loss,best_train_loss,'
    stest = 'best_test_epoch,current_test_loss,best_test_loss,'
    siou = ',,'*self.iou_num


    if isinstance(file, str):
      file = open(file, 'w')
    file.write(strain + siou + stest + siou + '\n')
    file.close()

  def write(self, file):
    '''
    self.best_iou = None
    self.best_loss = None
    self.best_epoch = None
    self.epoch = -1
    '''

    output_string = ''
    if (self.is_train):
      output_string += (str(self.epoch) + ',')
    #best_epoch, current loss, best loss, current/best iou,
    output_string += ('%d,%f,%f,'%(self.best_epoch, self.loss, self.best_loss))
    iou_len = self.iou.shape[0]
    for i in range(iou_len):
      output_string += ('%.05f,%.05f,' %(self.iou[i], self.best_iou[i]))

    if isinstance(file, str):
      file = open(file, 'a')
    if (not self.is_train):
      output_string += '\n'
    file.write(output_string)
    file.close()

def main():
  runtime_path = createTrainPath()

  args = parse_args()
  default_type = torch.float32
  torch.autograd.set_detect_anomaly(True)
  torch.set_default_dtype(default_type)

  epochs = args.epochs
  batch_size = args.batch_size
  lr = args.lr
  device = torch.device(args.device)
  seed = args.seed
  if (seed < 0):
    seed = int(time.time())
  print("random seed %d" % seed)
  random.seed(seed)
  torch.manual_seed(seed)
  if (False):
    dataset_path = '/home/moriarty/Projects/x.csv'
    image_path = '/home/moriarty/Datasets/coco/train2017'
    dataset = loadDataset(dataset_path)
    dataset = filter(dataset)
    dataset = Dataset(dataset, image_path)
  else:
    dataset_path = ['/home/moriarty/Datasets/wound/s1/labels',
                    '/home/moriarty/Datasets/wound/s2/labels',
                    '/home/moriarty/Datasets/wound/s3/labels',
                    '/home/moriarty/Datasets/wound/s4/labels']
    teset = createDatasetFromList([dataset_path[0]])
    trset = createDatasetFromList(dataset_path[1:])

  # size = len(dataset)
  # train_size = len(trset)

  if len(args.weights) == 0:
    model = TorchNet()  # create
    torch.save(model, runtime_path + '/init.pkl')
  else:
    print("loading weight from %s" %args.weights)
    model = torch.load(args.weights, map_location = device)

  model = model.to(device)
  compute_loss = Loss.MultipleLoss()

  # trset, teset,ddd = torch.utils.data.random_split(
  #   dataset,
  #   [train_size, size - train_size, 0],
  #   generator=torch.Generator().manual_seed(42))

  trloader = torch.utils.data.DataLoader(
    trset,
    batch_size=batch_size,
    shuffle=True,
    sampler=None,
    batch_sampler=None,
    num_workers=4,
    collate_fn=None,
    pin_memory=True,
    drop_last=False,
    timeout=0,
    worker_init_fn=None,
    prefetch_factor=4,
    persistent_workers=True)

  teloader = torch.utils.data.DataLoader(
    teset,
    batch_size=batch_size*2,
    shuffle=True,
    sampler=None,
    batch_sampler=None,
    num_workers=4,
    collate_fn=None,
    pin_memory=True,
    drop_last=False,
    timeout=0,
    worker_init_fn=None,
    prefetch_factor=4,
    persistent_workers=True)

  optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                               betas=(0.9, 0.99))
  train_loss = LossManager(True, 5)
  test_loss = LossManager(False)

  train_loss.init_file(runtime_path + '/result.csv')
  # te_result = test(model, teloader, compute_loss, -1, device)
  for epoch in range(epochs):
    begin = time.time()
    tr_result = train(model, trloader, optimizer, compute_loss, epoch, device)
    te_result = test(model, teloader, compute_loss, epoch, device)

    if (test_loss.update(te_result[0], te_result[1], epoch)):
      torch.save(model, os.path.join(runtime_path, 'best_test.pt'))
    if (train_loss.update(tr_result[0], tr_result[1], epoch)):
      torch.save(model, os.path.join(runtime_path, 'best_train.pt'))
    end = time.time()
    print('epoch %d finish, best(%d,%d), loss (%.04f,%.04f), '
          'iou (%.04f,%.04f) cost (%.02f+%.02f)=%.02f second' % (
      epoch, train_loss.best_epoch, test_loss.best_epoch,
      train_loss.loss, test_loss.loss,
      (train_loss.iou[-1]), (test_loss.iou[-1]),
      tr_result[2], te_result[2], end - begin))
    train_loss.write(runtime_path + '/result.csv')
    test_loss.write(runtime_path + '/result.csv')
    torch.save(model, os.path.join(runtime_path, 'last.pt'))

main()




