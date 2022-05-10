import torch
import numpy as np
import cv2
import os
import copy
import csv
import time
# import copy


def showAllImages(dataset, src = '/home/moriarty/Datasets/coco/train2017', dst = '/home/moriarty/Datasets/coco/draw'):
  '''
  @dataset:[[id, file_name, height, width, area, [segmentation]], ...,]
  '''
  for sample in dataset:
    sample = Sample(sample)
    img_path = os.path.join(src, sample.file_name())
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    image = draw(sample.segmentation(), image)
    img_path = os.path.join(dst, sample.file_name())

    cv2.circle(image, np.array(sample.segmentation()[0:2], dtype=np.int32), 5, (255, 0, 0), thickness=-1)
    print("draw circle position %s" %str(sample.segmentation()[0:2]))

    cv2.imwrite(img_path, image)

def fillPoly(polygons, image, color=(1, 1, 1)):
  polygons = [_.astype(np.int32) for _ in polygons]
  image = cv2.fillPoly(image, polygons, color)
  return image

def draw(polygon, image, color=(0, 0, 255)):
  if isinstance(polygon, list):
    polygon = np.array(polygon)
  if (polygon.ndim != 3):
    polygon = polygon.reshape((1, -1, 2))
  if (polygon.dtype != np.int32):
    polygon = polygon.astype(np.int32)

  return cv2.polylines(image, polygon, True, color, thickness=3)
  # cv2.imshow('image', image)
  # cv2.waitKey(0)

class Sample:
  def __init__(self, data:list):
    #@data: [id, file_name, height, width, area, [segmentation]]
    self.data = data[0:5]
    self.segmentations_ = [np.array(data[5:]).reshape(-1,2)]

  @staticmethod
  def preprocess_image(image):
    '''
    @image: read with cv2, [h, w, 3]
    '''
    # return (image.astype(dtype=np.float32).transpose(2, 0, 1) / (255.0 / 2)) - 1.0
    return (image.astype(dtype=np.float32).transpose(2, 0, 1) / (255.0))

  @staticmethod
  def draw_segmentation(image:np.ndarray, label:np.ndarray, threshold=0.5):
    '''
    @image: h, w, 3
    @label: h, w
    '''
    x = label > threshold
    image[x, 0] = 255
    print("num of segmentation is %d" %x.sum())
    return image
  @staticmethod
  def get_new_size(old_size, new_size):
    '''
    @return: size, np.ndarray
    get the proper size that old_size => new_size
    it follows the rull that: size[0] <= new_size[0] and size[1] <= new_size[1]
    at the same time, at least one is equal
    '''
    if not isinstance(new_size, np.ndarray):
      new_size = np.array(new_size)
    if not isinstance(old_size, np.ndarray):
      old_size = np.array(old_size)

    scale = new_size / old_size
    if (scale[0] < scale[1]):
      scale[1] = scale[0]
    else:
      scale[0] = scale[1]
    size = (scale[0] * old_size.astype(scale.dtype)).astype(np.int32)
    return size

  def index(self):
    return self.data[0]
  def file_name(self):
    return self.data[1]
  def image_size(self):
    #width, height
    return (self.data[3], self.data[2])
  def segment_area(self):
    return self.data[4]
  def segmentations(self):
    return self.segmentations_

  def create_label(self, _new_size, root):
    '''
    @_new_size:(resize current image to (width, height))
    @root: path to image
    '''
    path = os.path.join(root, self.file_name())
    image = cv2.imread(path, cv2.IMREAD_COLOR)

    old_size = np.array(self.image_size())
    new_size = np.array(_new_size)

    segmentations = copy.deepcopy(self.segmentations())

    #resize
    scale = new_size / old_size
    if (scale[0] < scale[1]):
      scale[1] = scale[0]
    else:
      scale[0] = scale[1]
    size = (scale[0] * old_size.astype(scale.dtype)).astype(np.int32)

    # print("resize image from, ", old_size, "=>", size, image.shape, self.file_name())
    _image = cv2.resize(image, (size[0], size[1]), interpolation=cv2.INTER_LINEAR)

    current_center = size / 2
    to_center = new_size / 2
    translate = (to_center - current_center).astype(np.int32)
    # print("current_center,", current_center, ",to_center,", to_center, 'translate', translate)

    image = np.zeros((_new_size[1], _new_size[0], 3), dtype=np.float32)
    # print('_image.shape ', _image.shape, image.shape)
    image[translate[1]:(translate[1] + _image.shape[0]), translate[0]:(translate[0] + _image.shape[1]), :] = _image

    label = np.zeros(image.shape[0:2], dtype=np.uint8)

    segmentations = [(_ * scale[0] + translate).astype(np.int32) for _ in segmentations]
    image = cv2.fillPoly(image, segmentations, (1.0, 1.0, 1.0))

    # _image = image.copy()
    # fillPoly(segmentation, _image, color=(1.0, 1.0, 1.0))

    image = Sample.preprocess_image(image)
    return image, np.expand_dims(label, axis=0).astype(np.float32)


class Dataset(torch.utils.data.Dataset):
  def __init__(self, dataset:list, path_to_image:str):
    self.image_size = (320, 320)
    self.path = copy.deepcopy(path_to_image)
    self.dataset = {}
    self.size = len(dataset)
    for idx, sample in enumerate(dataset):
      self.dataset[idx] = sample
  def __len__(self):
    return self.size
  def __getitem__(self, idx):
    item = Sample(self.dataset[idx])

    image, label = item.create_label(self.image_size, self.path)

    return [image, label]

def filter(dataset:list):
  '''
  only process samples that has one segmentation
  @dataset:[[id, file_name, height, width, area, [segmentation]], ...,]
  '''
  ndataset = []
  area_threshold = 0.1
  edge = 10
  for _ in dataset:
    #area ratio
    sample = Sample(_)
    shape = sample.image_size()
    img_area = shape[0] * shape[1]
    ratio = sample.segment_area() / img_area
    if (ratio < area_threshold):
      continue

    #do not too close to edge of image
    #only process samples that has one segmentation
    segmentation = sample.segmentations()[0]
    invalid = (segmentation[:, 0] < edge) | (segmentation[:, 0] > (shape[0] - edge)) | (segmentation[:, 1] < edge) | (segmentation[:, 0] > (shape[1] - edge))
    if (np.sum(invalid) > 2):
      continue
    ndataset.append(_)

  return ndataset

def saveDataset(dataset:list, path:str):
  '''
  @dataset:[[id, file_name, height, width, area, [segmentation]], ...,]
  '''

  file = open(path, 'w')
  writer = csv.writer(file)
  for sample in dataset:
    writer.writerow(sample)
  file.close()

def loadDataset(path:str):
  '''
  @return:[[id, file_name, height, width, area, [segmentation]], ...,]
  '''
  print("Loading Dataset from %s" %path)
  begin = time.time()
  file = open(path, 'r')
  reader = csv.reader(file)
  dataset = []
  for row in reader:
    sample = [int(row[0]), row[1], int(row[2]), int(row[3])]
    sample.extend([float(_) for _ in row[4:]])
    dataset.append(sample)
    # print("sample ", sample)
  file.close()
  cost = time.time() - begin
  print("Load dataset cost time %f sec" %cost)
  return dataset

def getImagesName(root:str):
  '''
  @root:path to images
  @return:[names]
  '''
  return os.listdir(root)
import json

def createDatasetFromLabelme(data:str):

  with open(data, "r") as f:
    row_data = json.load(f)
  # 读取每一条json数据


  for d in row_data:
    print(d)

