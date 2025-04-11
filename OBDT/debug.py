import os
import argparse

import numpy as np
import random
import tqdm

import torch
from pycocotools.coco import COCO
from torch.utils.data import DataLoader,Subset
import torch.optim as optim

from torchvision.datasets import CocoDetection
from torchvision import transforms

from datasets import CustomCocoDetection,CustomVOCDetection,AUGMENTATION_TRANSFORMS,DEBUG_TRANSFORMS,DEFAULT_TRANSFORMS
from model import MyNet,MiniDetector,ModifiedResnet
from utils import *
from config import *

from terminaltables import AsciiTable

from torchsummary import summary

import matplotlib.pyplot as plt
import matplotlib.patches as patches

'''
此文件用来展示数据标签或模型效果
'''


net=MyNet().to(device)
net.load_state_dict(torch.load("weights/target_recognition_step_2592000.pth"))
# torch.save(net.feature_layers.state_dict(), f"weights/pretrained/target_recognition_layers_step_{240000}.pth")
net.eval()

# 定义数据集的路径
image_folder = "C:/Datasets/VOC_2012"
annotation_file = "C:/Datasets/coco_2014/annotations/instances_val2014.json"
# image_folder = "C:/Datasets/coco_2014/train2014"
# annotation_file = "C:/Datasets/coco_2014/annotations/instances_train2014.json"

# coco = COCO(annotation_file)

voc_dataset_train = CustomVOCDetection(root=image_folder,image_set="val" ,transform=DEFAULT_TRANSFORMS)
my_image=torchvision.datasets.ImageFolder("./imgs",transform=transforms.Compose([
    transforms.CenterCrop(image_size),    # 中心裁剪
    transforms.ToTensor(),         # 转换为Tensor
]))

# 创建 DataLoader
subset_indices = torch.arange(0, subset_size)  # 选择前x个样本
subset = Subset(voc_dataset_train, subset_indices)
use_subset=False
# data_loader = DataLoader(subset if use_subset else voc_dataset_train, batch_size=2, shuffle=True, collate_fn=voc_dataset_train.collate_fn)
data_loader = DataLoader(my_image, batch_size=1, shuffle=False)

def show_dataset():
    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 80)]
    bbox_colors = random.sample(colors, 80)

    for batch_i, (imgs, targets) in enumerate(data_loader):
        print(len(imgs))
        img = np.array(imgs[0]).transpose((1, 2, 0))

        fig, ax = plt.subplots(1,figsize=(9,9))
        ax.imshow(img)
        for target in targets[0]:  # 逐个取批次中的第一张图中的每个目标
            cls, x1, y1, w, h = target[1], target[2], target[3], target[4], target[5]
            # 去归一化
            x1 *= image_size
            y1 *= image_size
            w *= image_size
            h *= image_size


            # color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            # Create a Rectangle patch
            bbox = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor="yellow", facecolor="none")

            # Add the bbox to the plot
            ax.add_patch(bbox)
            center=patches.Circle((x1+w/2,y1+h/2),3,edgecolor="pink")
            ax.add_patch(center)
            # Add label
            plt.text(
                x1,
                y1,
                s=f"bbox{cls}",
                color="white",
                verticalalignment="top",
                # bbox={"color": color, "pad": 0}
            )

        for i in range(7):
            for j in range(7):
                grid=patches.Rectangle((i*image_size/7, j*image_size/7), image_size/7, image_size/7, linewidth=1, edgecolor="black", facecolor="none")
                ax.add_patch(grid)
        plt.show()

def show_image_result(origin=False,grid_size=13):
    # bounding box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 80)]
    bbox_colors = random.sample(colors, 80)
    for batch_i, (imgs, targets) in enumerate(data_loader):
        img_for_input = imgs[0].numpy().transpose((1, 2, 0))
        output_list = net.forward_ret_pxywhcls_list(img_for_input, nms_for_each_group=False)
        print(len(output_list[0]))
        fig, ax = plt.subplots(1,figsize=(7,7))
        ax.imshow(img_for_input)

        if origin:
            for target in targets[0]:  # 逐个取批次中的第一张图中的每个目标
                cls, x1, y1, w, h = target[1], target[2], target[3], target[4], target[5]
                # 去归一化
                x1 *= image_size
                y1 *= image_size
                w *= image_size
                h *= image_size


                # color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor="yellow", facecolor="none")

                # Add the bbox to the plot
                ax.add_patch(bbox)
                center=patches.Circle((x1+w/2,y1+h/2),3,edgecolor="pink")
                ax.add_patch(center)
                # Add label
                plt.text(
                    x1,
                    y1,
                    s=f"bbox{cls}",
                    color="white",
                    verticalalignment="top",
                    # bbox={"color": color, "pad": 0}
                )

        for target in output_list[0]:  # 逐个取批次中的第一张图中的每个目标
            p, x1, y1, w, h ,cls= target[0], target[1], target[2], target[3], target[4],target[5]
            # 去归一化
            x1 *= image_size
            y1 *= image_size
            w *= image_size
            h *= image_size

            # Create a Rectangle patch
            bbox = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor="red", facecolor="none")
            # Add the bbox to the plot
            ax.add_patch(bbox)

            center = patches.Circle((x1 + w / 2, y1 + h / 2), 3, edgecolor="red")
            ax.add_patch(center)
            # Add label
            plt.text(
                x1,
                y1,
                s=f"bbox{int(cls)+1}: {p:.2f}",
                color="white",
                verticalalignment="top",
                # bbox={"color": color, "pad": 0}
            )




        # for i in range(grid_size):
        #     for j in range(grid_size):
        #         grid=patches.Rectangle((i*image_size/grid_size, j*image_size/grid_size), image_size/grid_size, image_size/grid_size, linewidth=1, edgecolor="black", facecolor="none")
        #         ax.add_patch(grid)



        plt.show()

# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    # show_dataset()
    show_image_result()

