from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import glob
import random
import os
import warnings
import numpy as np
from PIL import Image
from PIL import ImageFile
from torchvision.datasets import CocoDetection,VOCDetection
from torchvision import datasets, transforms
from utils import xywh2xyxy_np
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import imgaug.augmenters as iaa
from config import *



ImageFile.LOAD_TRUNCATED_IMAGES = True

# 数据导入与增强

class ImgAug(object):
    def __init__(self, augmentations=[]):
        self.augmentations = augmentations

    def __call__(self, data):
        # Unpack data
        img, boxes = data

        # Convert xywh to xyxy
        boxes = np.array(boxes)
        boxes[:, 1:] = xywh2xyxy_np(boxes[:, 1:])

        # Convert bounding boxes to imgaug
        bounding_boxes = BoundingBoxesOnImage(
            [BoundingBox(*box[1:], label=box[0]) for box in boxes],
            shape=img.shape)

        # Apply augmentations
        img, bounding_boxes = self.augmentations(
            image=img,
            bounding_boxes=bounding_boxes)

        # Clip out of image boxes
        bounding_boxes = bounding_boxes.clip_out_of_image()

        # Convert bounding boxes back to numpy
        boxes = np.zeros((len(bounding_boxes), 5))
        for box_idx, box in enumerate(bounding_boxes):
            # Extract coordinates for unpadded + unscaled image
            x1 = box.x1
            y1 = box.y1
            x2 = box.x2
            y2 = box.y2

            # Returns (x, y, w, h)
            boxes[box_idx, 0] = box.label
            boxes[box_idx, 1] = x1
            boxes[box_idx, 2] = y1
            boxes[box_idx, 3] = (x2 - x1)
            boxes[box_idx, 4] = (y2 - y1)

        return img, boxes


# 归一化01
class RelativeLabels(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, boxes = data
        h, w, _ = img.shape
        boxes[:, [1, 3]] /= w
        boxes[:, [2, 4]] /= h
        return img, boxes

# 去归一化
class AbsoluteLabels(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, boxes = data
        h, w, _ = img.shape
        boxes[:, [1, 3]] *= w
        boxes[:, [2, 4]] *= h
        return img, boxes

# 填充成正方形
class PadSquare(ImgAug):
    def __init__(self, ):
        super().__init__()
        self.augmentations = iaa.Sequential([
            iaa.PadToAspectRatio(
                1.0,
                position="center-center").to_deterministic()
        ])


class ToTensor(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, boxes = data
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(img)

        bb_targets = torch.zeros((len(boxes), 5))
        bb_targets[:, :] = transforms.ToTensor()(boxes)

        return img, bb_targets


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        img, boxes = data
        img = F.interpolate(img.unsqueeze(0), size=self.size, mode="nearest").squeeze(0)
        return img, boxes


class Augment(ImgAug):
    def __init__(self, ):
        super().__init__()
        self.augmentations = iaa.Sequential([
            iaa.Sharpen((0.0, 0.5)),
            iaa.Affine(rotate=(-3,3),translate_percent=(-0.05, 0.05), scale=(0.9, 1.1)),
            iaa.AddToBrightness((-60, 60)),
            iaa.AddToHue((-210, 210)),
            iaa.GaussianBlur(sigma=(0, 1.0)),
            iaa.Fliplr(0.5),
        ])

class DebugAugment(ImgAug):
    def __init__(self, ):
        super().__init__()
        self.augmentations = iaa.Sequential([
            iaa.Sharpen((0.0, 0.5)),
            # iaa.Affine(rotate=(-4,4),translate_percent=(-0.3, 0.3), scale=(0.75, 1.25)),
            iaa.AddToBrightness((-60, 60)),
            iaa.AddToHue((-210, 210)),
            iaa.GaussianBlur(sigma=(0, 1.0)),
            iaa.Fliplr(0.5),
        ])


DEFAULT_TRANSFORMS = transforms.Compose([
    PadSquare(),
    RelativeLabels(),
    ToTensor(),
    Resize((image_size,image_size)),
])

AUGMENTATION_TRANSFORMS = transforms.Compose([
    Augment(),
    PadSquare(),
    RelativeLabels(),
    ToTensor(),
    Resize((image_size,image_size)),
])

DEBUG_TRANSFORMS = transforms.Compose([
    DebugAugment(),
    PadSquare(),
    RelativeLabels(),
    ToTensor(),
    Resize((image_size,image_size)),
])


# def pad_to_square(img, pad_value):
#     c, h, w = img.shape
#     dim_diff = np.abs(h - w)
#     # (upper / left) padding and (lower / right) padding
#     pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
#     # Determine padding
#     pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
#     # Add padding
#     img = F.pad(img, pad, "constant", value=pad_value)
#
#     return img, pad
#
#
# def resize(image, size):
#     image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
#     return image




class CustomCocoDetection(CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None,ids=None):
        super(CustomCocoDetection, self).__init__(root, annFile)
        self.transform = transform
        self.target_transform = target_transform
        if ids is not None:
            self.ids=ids
        self.batch_count=0 #这个值每次调用getitem就会加1，直到调用collate_fn清零，用来指示当前getitem读取的图片是一个batch中的第几张

    def __getitem__(self, index):
        img, target = super(CustomCocoDetection, self).__getitem__(index)
        img = np.array(img, dtype=np.uint8)
        bboxes = np.array([obj['bbox'] for obj in target])
        categories= np.array([obj['category_id'] for obj in target])
        # 把类别标签添加到bboxe前面，成为5维
        categories = categories[:, np.newaxis]

        # 如果标签与框不符，添加错误标记-1
        try:
            temp=np.hstack((categories, bboxes))
        except:
            temp=np.full((1, 5),-1)

        if self.transform is not None:
            img,temp=self.transform((img, temp))
        # 将图像批次id添加到前面成为6维
        try:
            arr = torch.full((bboxes.shape[0],1), self.batch_count)
            temp = torch.hstack((arr, temp))
        except:
            temp=torch.full((1,6),-1)

        self.batch_count += 1
        return img, temp

    # def _filter_imgs_by_category(self, category_ids):
    #     # 筛选出包含指定类别的图像
    #     img_ids = coco.getImgIds(catIds=category_ids)
    #     return img_ids

    def collate_fn(self,batch):
        self.batch_count=0
        filtered_batch=[]
        for data in batch:
            # 检查维数是否正确
            if data[1].size(0)>0:
                if data[1][0][0]!=-1:
                    filtered_batch.append(data)
        return tuple(zip(*filtered_batch))


class CustomImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except (OSError, IOError) as e:
            print(f"Warning: Skipping corrupted image at index {index}")
            return None



class CustomVOCDetection(VOCDetection):
    def __init__(self, root, transform=None, target_transform=None,ids=None,image_set=None):
        super(VOCDetection, self).__init__(root,image_set=image_set)
        self.transform = transform
        self.target_transform = target_transform
        if ids is not None:
            self.ids=ids
        self.batch_count=0 #这个值每次调用getitem就会加1，直到调用collate_fn清零，用来指示当前getitem读取的图片是一个batch中的第几张
        self.id_dict={'person':1,'bird':2,'cat':3,'cow':4,'dog':5,'horse':6,'sheep':7,'aeroplane':8,'bicycle':9,'boat':10,'bus':11, 'car':12, 'motorbike':13,'train':14,'bottle':15,'chair':16,'diningtable':17,'pottedplant':18,'sofa':19,'tvmonitor':20}

    def __getitem__(self, index):
        img, target = super(CustomVOCDetection, self).__getitem__(index)
        img = np.array(img, dtype=np.uint8)
        tgs=target['annotation']['object']
        bbox=[]
        catg=[]
        for obj in tgs:
            bbox.append(([int(obj['bndbox']['xmin']),int(obj['bndbox']['ymin']),int(obj['bndbox']['xmax']),int(obj['bndbox']['ymax'])]))
            id=int(self.id_dict[obj['name']])
            catg.append([id])
        bboxes = np.array(bbox)#xyxy
        bboxes[...,2]=bboxes[...,2]-bboxes[...,0]
        bboxes[..., 3] = bboxes[..., 3] - bboxes[..., 1]
        categories= np.array(catg)
        # 把类别标签添加到bboxe前面，成为5维
        # categories = categories[:, np.newaxis]

        # 如果标签与框不符，添加错误标记-1
        try:
            temp=np.hstack((categories, bboxes))
        except:
            temp=np.full((1, 5),-1)

        if self.transform is not None:
            img,temp=self.transform((img, temp))
        # 将图像批次id添加到前面成为6维
        try:
            arr = torch.full((bboxes.shape[0],1), self.batch_count)
            temp = torch.hstack((arr, temp))
        except:
            temp=torch.full((1,6),-1)

        self.batch_count += 1
        return img, temp

    # 校对batch中是否有图片
    def collate_fn(self,batch):
        self.batch_count=0
        filtered_batch=[]
        for data in batch:
            # 检查维数是否正确
            if data[1].size(0)>0:
                if data[1][0][0]!=-1:
                    filtered_batch.append(data)
        return tuple(zip(*filtered_batch))