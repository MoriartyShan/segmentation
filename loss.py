import torch



def IOU(output, target):
  output = output > 0.8
  target = target > 0.8
  intersection = output & target
  union = output | target
  return intersection.sum() / union.sum()



class MultipleLoss:
  def __init__(self):
    self.BCE = torch.nn.BCELoss()
    # self.BCEWithLogitsLoss = torch.nn.BCEWithLogitsLoss()
  def __call__(self, output, target):
    bce = self.BCE(output, target)
    iou = IOU(output, target)
    return bce, iou