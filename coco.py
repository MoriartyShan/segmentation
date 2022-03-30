import time
import copy
from pycocotools.coco import COCO
import cv2

def getSegmentPersonFromCoco(
    ann_train_file_path:str= '/home/moriarty/Datasets/coco/'
      'annotations/instances_train2017.json'):
  '''
  @return: [[id, file_name, height, width, area, [segmentation]], ...,]
  '''
  coco_train = COCO(ann_train_file_path)
  singleTypeDict = singleTypeFilter(coco_train)

  dataset = translateCocoDict(singleTypeDict)

  return dataset

def singleTypeFilter(coco_train, type=1):
  '''
  get samples whose type id is 1, and one image contains only one this type object
  @type id check
  # for categorie in categories:
  #   print(categorie)
  '''
  begin = time.time()
  # coco_train.info()
  # coco_train = copy.deepcopy(_coco_train)

  coco_images = coco_train.dataset['images']
  annotations = coco_train.dataset['annotations']
  categories = coco_train.dataset['categories']

  image_lables = {}

  # for categorie in categories:
  #   print(categorie)

  for image in coco_images:
    id = image['id']
    image_lables[id] = copy.deepcopy(image)

  for annotation in annotations:

    if (annotation['category_id'] != type):
      continue

    id = annotation['image_id']
    if (image_lables.__contains__(id)):
      img = image_lables[id]
      if (img.__contains__('annotation')):
        img['annotation'].append(annotation)
      else:
        img['annotation'] = [annotation]
    else:
      print("do not contains image %d" % (id))

  filtered = {}
  size = len(image_lables)
  current = 0
  for id, info in image_lables.items():
    current+=1
    if not info.__contains__('annotation'):
      continue
    if (len(info['annotation']) != 1):
      continue
    annotation = info['annotation'][0]
    # print("has ann")
    if (annotation['iscrowd'] == 1):
      # RLE
      print(annotation['segmentation'])
      continue
    image = cv2.imread('/home/moriarty/Datasets/coco/train2017/' + info['file_name'])
    if (image.size == 0):
      print("image [%s] is empty" % (info['file_name']))
      continue
    # print("has ann %d, %s" %(id, str(info)))
    print("preprocessed data %d/%d, %.2f" %(current, size, 100.0 * current/size))

    # img = draw(np.array(annotation['segmentation']), image)
    # cv2.imwrite("newimg.png", img)
    # break
    # if (not annotation.__contains__('segmentation')):
    #   print(annotation)
    #   break
    #   continue
    filtered[id] = info
  print("contains single per image %d" % len(filtered))
  print("cost time %f" %(time.time() - begin))
  return filtered

def translateCocoDict(datas:dict):
  '''
  @datas: a dict, {id(int), infos}
  @infos: a dict, has keys ['license', 'file_name', 'coco_url', 'height', 'width', 'date_captured', 'flickr_url', 'id', 'annotation']
  @annotation: a list, typically contains only one object
  @annotation[0]: a dict, has keys ['segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'category_id', 'id']
  @annotation[0][segmentation]: [[x0, y0, x1, y1, ..., ]]

  @return: [[id, file_name, height, width, area, [segmentation]], ...,]
  '''
  begin = time.time()
  print("Translating Coco Datasets...")
  translated = []
  for id, infos in datas.items():
    annotation = infos['annotation'][0]
    info = [id, infos['file_name'], infos['height'], infos['width'], annotation['area']]
    info.extend(annotation['segmentation'][0])
    translated.append(info)
  print("Translate finish, cost %f sec" %(time.time() - begin))
  return translated