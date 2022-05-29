import torch
import models.common as common
import models.yolo as yolo

import torchvision.models.segmentation as torch_segmentation


class FullConvNet(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.channels = [3, 8, 16, 32, 64, 64, 1]
    self.kernels = [7, 5, 3, 3, 3, 3]
    convs = []
    num = len(self.kernels)
    for i in range(num):
      convs.append(common.Conv(self.channels[i], self.channels[i+1], self.kernels[i]))
    self.conv = torch.nn.Sequential(*convs)
  def forward(self, x):
    return self.conv(x)

class TorchNet(torch.nn.Module):
  def __init__(self):
    super().__init__()
    # self.model = torch_segmentation.deeplabv3_resnet50(num_classes=1)
    # self.model = yolo.Model('/home/moriarty/Projects/segmentation/models/Simple.yaml')
    self.model = torch_segmentation.lraspp_mobilenet_v3_large(num_classes=1)
    # self.model = torch_segmentation.deeplabv3_mobilenet_v3_large(num_classes=1)
    # self.act = torch.nn.Sigmoid()
    self.act = torch.nn.Tanh()
  def forward(self, x):
    # return self.act(self.model(x)['out'])
    if isinstance(self.model, yolo.Model):
      x = self.model(x)
    else:
      x = self.model(x)['out']
    # print("x range %f, %f" %(torch.max(x), torch.min(x)))
    if isinstance(self.act, torch.nn.Sigmoid):
      return self.act(x)
    elif isinstance(self.act, torch.nn.Tanh):
      return (self.act(x) + 1.0)/2.0

    return self.act(x)
