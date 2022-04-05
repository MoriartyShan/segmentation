import torch
import time
def test_time(_model, _x):
  begin = time.time()
  y = _model(_x)
  print('sum y %f' %y['out'][0,0,0,0])
  print("cost time %f" %(time.time() - begin))

def IOU(output, target, threshold=0.5):
  output = output > threshold
  target = target > threshold
  intersection = output & target
  union = output | target
  return intersection.sum() / union.sum()



class MultipleLoss:
  def __init__(self):
    self.BCE = torch.nn.BCELoss()
    # self.BCEWithLogitsLoss = torch.nn.BCEWithLogitsLoss()
  def __call__(self, output, target):
    bce = self.BCE(output, target)
    iou = []
    for i in range(5):
      iou.append(IOU(output, target, i * 0.09 + 0.5))
    iou = torch.stack(iou, dim=0)
    return bce, iou