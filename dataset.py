import torch
import numpy as np
import cv2
import os
import copy
import csv
import time
from torch.types import Union
# import copy
_image_size=(320, 320)

import filecmp

def filterSameFiles(pathA:str, pathB:str):
  namesA = getFilesName(pathA)
  namesB = getFilesName(pathB)
  theSame = []
  for A in namesA:
    pA = os.path.join(pathA, A)
    for B in namesB:
      pB = os.path.join(pathB, B)
      if (filecmp.cmp(pA, pB)):
        theSame.append((A, B))
  return theSame

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
  def __init__(self, data:Union[list, dict], root:str):
    '''
    @data: list [id, file_name, height, width, area, [segmentation]]
           dict[flags, version, shapes, imagePath, imageData, imageHeight, imageWidth]
    @self.data: [id, file_name, height, width, [area], [segmentation]]
    '''

    if isinstance(data, list):
      self.data = data[0:5]
      self.data[-1] = [self.data[-1]]
      self.segmentations_ = [np.array(data[5:]).reshape(-1,2)]
      self.path = os.path.join(root, self.file_name())
    elif isinstance(data, dict):
      self.data = [
        0,
        os.path.basename(data['imagePath']),
        data['imageHeight'],
        data['imageWidth'],
        []]
      self.path = os.path.join(root, data['imagePath'])
      self.segmentations_ = []
      shapes = data['shapes']
      for shape in shapes:
        poly = np.array(shape['points'], dtype=np.float32)
        self.segmentations_.append(poly)
        self.data[-1].append(cv2.contourArea(poly))

    else:
      print("error init sample type %s" %str(type(data)))
      exit(0)

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

  def create_label(self, _new_size = None):
    '''
    @_new_size:(resize current image to (width, height))
    @root: path to image
    '''
    image = cv2.imread(self.path, cv2.IMREAD_COLOR)

    old_size = np.array(self.image_size())

    if (_new_size is None):
      _new_size = old_size
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
    resized_image = cv2.resize(image, (size[0], size[1]), interpolation=cv2.INTER_LINEAR)
    # print("resized_image image type ", image.dtype)

    current_center = size / 2
    to_center = new_size / 2
    translate = (to_center - current_center).astype(np.int32)
    # print("current_center,", current_center, ",to_center,", to_center, 'translate', translate)

    new_size_image = np.zeros((_new_size[1], _new_size[0], 3), dtype=np.float32)
    # print('resized_image.shape ', resized_image.shape, new_size_image.shape)
    new_size_image[translate[1]:(translate[1] + resized_image.shape[0]), translate[0]:(translate[0] + resized_image.shape[1]), :] = resized_image

    segmentations = [(_ * scale[0] + translate).astype(np.int32) for _ in segmentations]
    label = cv2.fillPoly(np.zeros(new_size_image.shape[0:2], dtype=np.uint8), segmentations, (1.0, 1.0, 1.0))
    preprocessed_image = Sample.preprocess_image(new_size_image)
    return preprocessed_image, np.expand_dims(label, axis=0).astype(np.float32), new_size_image
  def show_label(self, size = None):
    preprocessed_image, label, image = self.create_label(size)
    image = image.astype(np.uint8)
    draw = Sample.draw_segmentation(image, label[0], 0.5)
    return draw

class Dataset(torch.utils.data.Dataset):
  def __init__(self, dataset:list, path_to_image:str):
    self.image_size = _image_size
    self.path = copy.deepcopy(path_to_image)
    self.size = len(dataset)

    samples = []
    for data in dataset:
      samples.append(Sample(data, self.path))

    self.dataset = np.array(samples, dtype=np.object)
    print('shape ', self.dataset.shape)

    for idx, sample in enumerate(dataset):
      self.dataset[idx] = Sample(sample, self.path)

  def __len__(self):
    return self.size
  def __getitem__(self, idx):
    item = self.dataset[idx]
    preprocessed_image, label, image = item.create_label(self.image_size)
    return [preprocessed_image, label]

class CombinedDataset(torch.utils.data.Dataset):
  def __init__(self, datasets:[torch.utils.data.Dataset]):
    self.datasets = datasets
    self.dataset_num = len(self.datasets)

    self.sizes = np.zeros(self.dataset_num+1, dtype=np.int)
    for i in range(0, self.dataset_num):
      self.sizes[i+1] = len(self.datasets[i]) + self.sizes[i]
    print("use combined dataset %s " %(str(self.sizes)))
  def __len__(self):
    return self.sizes[-1]

  def __getitem__(self, idx):
    for i in range(self.dataset_num):
      if (idx < self.sizes[i+1]):
        this_idx = idx - self.sizes[i]
        return self.datasets[i].__getitem__(this_idx)
    print("invalid input index %d, %d" %(idx, self.sizes[-1]))
    exit(0)

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

def getFilesName(root:str):
  '''
  @root:path to images
  @return:[names]
  '''
  return os.listdir(root)
import json

def loadDatasetFromLabelme(data_path:str):
  '''
  return a list contains Dict item
  @data_path: path to dir contains .json file, which created by labelme
  version
  flags
  shapes
  imagePath
  imageData
  imageHeight
  imageWidth
  @return: list of Dict items, do not contain imageData
  '''
  dataset = []
  image_names = getFilesName(data_path)
  for name in image_names:
    json_path = os.path.join(data_path, name)
    with open(json_path, "r") as f:
      raw_data = json.load(f)
      raw_data.pop('imageData')
      dataset.append(raw_data)


  dataset = Dataset(dataset, data_path)
  return dataset

def createDatasetFromList(datalist:[str]):
  datasets = []
  for data in datalist:
    datasets.append(loadDatasetFromLabelme(data))
  return CombinedDataset(datasets)

