from dataset import *
dataset = loadDatasetFromLabelme('/home/moriarty/Datasets/wound/s1/labels')
sample = dataset.get_sample(100)
for i in range(9):
  show = sample.show_label()
  cv2.imwrite("show_" + str(i) + ".png", show)
