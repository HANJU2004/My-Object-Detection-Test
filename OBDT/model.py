from __future__ import division

import os
from itertools import chain
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights

from config import *
from my_lab.layers import *
from utils import nms
import config

# from utils_lili.parse_config import parse_model_config
# from utils_lili.utils_lili import weights_init_normal



# TODO:全部stride instead of maxpool，lr递减


# 网络输入batch*3*448*448,输出batch*7*7*5*(5+100)
class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        # pretrain,train,inference

        self.name = "target_recognition"

        # 按传播顺序定义输出层，每层大小，及每层的锚框
        self.num_yolo_layers=3
        self.grid=[13,26,52]
        self.split_anchor_list=np.array([[[116,90],[156,198],[373,326]], [[30,61],[62,45],[59,119]], [[10,13],[16,30],[33,23]]],dtype=np.float32)
        for i in range(len(self.grid)):
            self.split_anchor_list[i]=self.split_anchor_list[i]*self.grid[i]/416
        self.split_anchor_list=self.split_anchor_list.tolist()


        if self.num_yolo_layers!=len(self.split_anchor_list):
            raise ValueError("layers count does not correspond to anchors count")

        self.layer1 = nn.Sequential(
            # input 416
            # nn.Conv2d(3, 64, 7, 2, 4),
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.layer2 = nn.Sequential(
            # 104
            MultiScaleConv(64, 64),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(),
            nn.Conv2d(192, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, 1, 1, 0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, 1, 1, 0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.15),
        #     接到yolo3
        )
        self.layer3 = nn.Sequential(
            # 52
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, 1, 1, 0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, 1, 1, 0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, 1, 1, 0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.15),
        #     接到up2
        )
        self.layer4 = nn.Sequential(
            # 26
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 256, 1, 1, 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 256, 1, 1, 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 256, 1, 1, 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.15),
        #     接到up1
        )
        self.layer5 = nn.Sequential(
            # 13
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 512, 1, 1, 0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 512, 1, 1, 0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 512, 1, 1, 0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 512, 1, 1, 0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Dropout2d(0.15),
        )

        # 每个yolo层使用的anchor不同
        self.yolo1=nn.Sequential(
            # 最大的3个anchor
            nn.Conv2d(512, 75, 1, 1, 0),
            Reshape(-1,13,13,3,25),
            SelectiveSigmoid(1,3),
            SelectiveSoftplus(3,5),
        )

        self.up1=nn.Sequential(
            # 13
            nn.Conv2d(1024, 512, 1, 1, 0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Upsample((26,26),mode="bilinear"),
            nn.Conv2d(512, 256, 1, 1, 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 256, 1, 1, 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 256, 1, 1, 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )

        self.yolo2 = nn.Sequential(
            nn.Conv2d(512, 75, 1, 1, 0),
            Reshape(-1, 26, 26, 3, 25),
            SelectiveSigmoid(1, 3),
            SelectiveSoftplus(3, 5),
        )

        self.up2 = nn.Sequential(
            # 26
            nn.Conv2d(512, 256, 1, 1, 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Upsample((52, 52), mode="bilinear"),
            nn.Conv2d(256, 128, 1, 1, 0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, 1, 1, 0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, 1, 1, 0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )

        self.yolo3 = nn.Sequential(
            nn.Conv2d(256, 75, 1, 1, 0),
            Reshape(-1, 52, 52, 3, 25),
            SelectiveSigmoid(1, 3),
            SelectiveSoftplus(3, 5),
        )



        self.avg_pool=nn.AdaptiveAvgPool2d((13,13))
        self.activate_for_confidence = nn.Sigmoid()
        self.activate_for_xy = nn.Sigmoid()
        self.activate_for_wh = nn.Softplus()
        self.activate_for_class = nn.Sigmoid()
        self.activate_for_pretrain = nn.Softmax(dim=1)


    def forward(self, x):

        l1_out = self.layer1(x)
        l2_out=self.layer2(l1_out)
        l3_out=self.layer3(l2_out)
        l4_out=self.layer4(l3_out)
        l5_out=self.layer5(l4_out)
        yl1_out=self.yolo1(l5_out)#13*13*3*(5+20)
        up1_out=self.up1(torch.cat([l5_out,l4_out],dim=1))
        yl2_out=self.yolo2(torch.cat([up1_out,l3_out],dim=1))#26*26*3*(5+20)
        up2_out=self.up2(torch.cat([up1_out,l3_out],dim=1))
        yl3_out=self.yolo3(torch.cat([up2_out,l2_out],dim=1))#52*52*3*(5+20)

        # p，x，y, w,h,cls激活映射



        return [yl1_out,yl2_out,yl3_out]


    # 返回符合条件的pxywhcls列表，暂时无法批量操作
    def forward_ret_pxywhcls_list(self, x0, nms_for_each_group=False):

        x = x0.transpose((2, 0, 1))
        x = torch.tensor(x, dtype=torch.float).to(device)
        if x.shape != (3, image_size, image_size):
            # x=F.pad()
            x = F.interpolate(x.unsqueeze(0), size=(image_size, image_size), mode="nearest")
        else:
            x=x.unsqueeze(0)

        output_for_all_layers=self.forward(x)
        output_list=[]

        for count in range(self.num_yolo_layers):
            anchor_size_list=self.split_anchor_list[count]
            output = output_for_all_layers[count].squeeze().detach().cpu()  # shape(7,7,5,5+100)去掉batch
            output[...,0]=self.activate_for_confidence(output[...,0])
            output[...,5:]=self.activate_for_class(output[...,5:])

            for i in range(self.grid[count]):
                for j in range(self.grid[count]):
                    for k in range(3):
                        if output[i,j,k,0]>=0.3:#置信度大于阈值时
                            #切片出当前数据并转换格式(相对于整体图像的归一化xywh)
                            temp=np.array(output[i,j,k])
                            temp[1]=(temp[1]+i)/self.grid[count]
                            temp[2]=(temp[2]+j)/self.grid[count]
                            temp[3]=temp[3]*anchor_size_list[k][0]/(self.grid[count])
                            temp[4]=temp[4]*anchor_size_list[k][1]/(self.grid[count])
                            class_id=np.argmax(temp[5:])
                            x1 = temp[1] - temp[3] / 2
                            y1 = temp[2] - temp[4] / 2
                            w = temp[3]
                            h = temp[4]
                            result=[temp[0],x1,y1,w,h,class_id]
                            if temp[5:][class_id]>=0.2:
                                output_list.append(result)

        if output_list==[]:
            return [[]]

        if len(output_list)>=200:
            output_list=output_list[0:200]

        # 创建一个字典，用于存储不同索引的分组
        groups = {}
        for sublist in output_list:
            # 找到子列表中最大元素的索引
            cls = sublist[-1]
            # 如果该索引不在字典中，则创建一个新的列表
            if cls not in groups:
                groups[cls] = []
            # 将子列表加入相应索引的分组中
            groups[cls].append(sublist)

        disposed_output_list=[]

        if nms_for_each_group:
            for element in groups:
                disposed_output_list.append(nms(groups[element]))
        else:
            disposed_output_list.append(nms(output_list))

        # 返回一个需要画出的框的列表[[[p,x,y,w,h,cls1],[p,x,y,w,h,cls1]],[[p,x,y,w,h,cls2],[p,x,y,w,h,cls2]]]
        return disposed_output_list


class ModifiedResnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.name="res_yolo"
        self.num_yolo_layers = 1
        self.grid = [13]

        self.split_anchor_list = np.array(
            [[[156, 198], [373, 326], [62, 45]]],
            dtype=np.float32)

        self.split_anchor_list = self.split_anchor_list * 13 / 416
        self.split_anchor_list = self.split_anchor_list.tolist()

        self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.avgpool=nn.Sequential()
        self.model.fc=nn.Sequential(
            nn.Unflatten(1,(2048,13,13)),
            nn.Conv2d(2048,75,1,1,0),
        )
        # print(self.model)
        # torch.save(self.model.state_dict(),"weights/pretrained/Resnet50_yolo.pth")
        self.activate_for_confidence = nn.Sigmoid()
        self.activate_for_xy = nn.Sigmoid()
        self.activate_for_wh = nn.Softplus()
        self.activate_for_class = nn.Sigmoid()

    def freeze(self):
        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False

        # 只训练最后层
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self,x):
        x=self.model(x).reshape(-1,13,13,3,25)
        # x = x.reshape(-1, 13, 13, len(self.split_anchor_list[0]), 5 + 20)  # 重塑为标准输出格式
        # p，x，y, cls映射到0，1
        # x[:, :, :, :, 0] = (self.activate_for_confidence(x[:, :, :, :, 0])+1)/2
        x[:, :, :, :, 1:3] = self.activate_for_xy(x[:, :, :, :, 1:3])
        # x[:, :, :, :, 5:]=self.activate_for_class(x[:, :, :, :, 5:])
        # w，h映射到0，+INF
        x[:, :, :, :, 3:5] = (x[:, :, :, :, 3:5].exp() + 1).log()
        return [x]


    def forward_ret_pxywhcls_list(self, x0, nms_for_each_group=False,ret_origin_array=False):
        x = x0.transpose((2, 0, 1))
        x = torch.tensor(x, dtype=torch.float).to(device)
        if x.shape != (3, image_size, image_size):
            # x=F.pad()
            x = F.interpolate(x.unsqueeze(0), size=(image_size, image_size), mode="nearest")
        else:
            x=x.unsqueeze(0)
        output = self.forward(x)[0].squeeze(0).detach().cpu()  # shape(7,7,5,5+100)去掉batch
        output[...,0]=self.activate_for_confidence(output[...,0])
        output[...,5:]=self.activate_for_class(output[...,5:])
        grid_size=output.shape[0]

        output_list=[]
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(output.shape[2]):
                    if output[i,j,k,0]>=0.4:#置信度大于阈值时
                        #切片出当前数据并转换格式(相对于整体图像的归一化xywh)
                        temp=np.array(output[i,j,k])
                        temp[1]=(temp[1]+i)/grid_size
                        temp[2]=(temp[2]+j)/grid_size
                        temp[3]=temp[3]*self.split_anchor_list[0][k][0]/grid_size
                        temp[4]=temp[4]*self.split_anchor_list[0][k][1]/grid_size
                        class_id=np.argmax(temp[5:])
                        x1 = temp[1] - temp[3] / 2
                        y1 = temp[2] - temp[4] / 2
                        w = temp[3]
                        h = temp[4]
                        result=[temp[0],x1,y1,w,h,class_id]
                        if output[i,j,k,0]*temp[5:][class_id]>=0.05:
                            output_list.append(result)

        if output_list==[]:
            if ret_origin_array:
                return [[]], output.numpy()
            else:
                return [[]]


        # 创建一个字典，用于存储不同索引的分组
        groups = {}
        for sublist in output_list:
            # 找到子列表中最大元素的索引
            cls = sublist[-1]
            # 如果该索引不在字典中，则创建一个新的列表
            if cls not in groups:
                groups[cls] = []
            # 将子列表加入相应索引的分组中
            groups[cls].append(sublist)

        disposed_output_list=[]

        if nms_for_each_group:
            for element in groups:
                disposed_output_list.append(nms(groups[element]))
        else:
            disposed_output_list.append(nms(output_list))

        # 返回一个需要画出的框的列表[[[p,x,y,w,h,cls1],[p,x,y,w,h,cls1]],[[p,x,y,w,h,cls2],[p,x,y,w,h,cls2]]]
        if ret_origin_array:
            return disposed_output_list,output.numpy()
        else:
            return disposed_output_list



class Classifier(nn.Module):
    def __init__(self, mode="train"):
        super().__init__()
        self.name="pretrain_classifier"
        self.layer1 = nn.Sequential(
            # input 416
            # nn.Conv2d(3, 64, 7, 2, 4),
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.layer2 = nn.Sequential(
            # 104
            MultiScaleConv(64, 64),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(),
            nn.Conv2d(192, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, 1, 1, 0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, 1, 1, 0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.01),
            #     接到yolo3
        )
        self.layer3 = nn.Sequential(
            # 52
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, 1, 1, 0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, 1, 1, 0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, 1, 1, 0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.01),
            #     接到up2
        )
        self.layer4 = nn.Sequential(
            # 26
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 256, 1, 1, 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 256, 1, 1, 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 256, 1, 1, 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.01),
            #     接到up1
        )
        self.layer5 = nn.Sequential(
            # 13
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 512, 1, 1, 0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 512, 1, 1, 0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 512, 1, 1, 0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 512, 1, 1, 0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Dropout2d(0.01),
        )

        self.avg_pool=nn.AdaptiveAvgPool2d((7,7))

        self.output_for_pretrain = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,1000),
            # nn.Softmax(dim=1),
        )






    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x).reshape(-1, 25088)
        # x = self.avg_pool(x).reshape(-1, 25088)
        x = self.output_for_pretrain(x)
        return x



class MiniDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.name="mini_detector"

        # self.split_anchor_list = np.array(
        #     [ [[156, 198], [373, 326],  [62, 45]]])

        # self.split_anchor_list= self.split_anchor_list * 13 / 416
        # self.split_anchor_list = self.split_anchor_list.tolist()
        self.split_anchor_list=[[[4.875, 6.1875], [11.65625, 10.1875], [1.9375, 1.40625]]]

        self.feature_layers = nn.Sequential(
            # nn.Conv2d(3, 64, 7, 2, 4),
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),


            MultiScaleConv(64,64),
            nn.LeakyReLU(),
            nn.Conv2d(192, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, 1, 1, 0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),

            nn.Dropout2d(0.3),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, 1, 1, 0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),

            nn.Dropout2d(0.3),

            #repeat
            nn.Conv2d(256, 128, 1, 1, 0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, 1, 1, 0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),

            nn.Dropout2d(0.3),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 256, 1, 1, 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),

            nn.Dropout2d(0.3),

            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 512, 1, 1, 0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 512, 1, 1, 0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),

            nn.Dropout2d(0.3),
        )


        self.yolo=nn.Sequential(
            # nn.Conv2d(1024, 125, 1, 1, 0),
            nn.Conv2d(1024, 75, 1, 1, 0),
        )

        self.avg_pool=nn.AdaptiveAvgPool2d((7,7))




        self.activate_for_confidence = nn.Sigmoid()
        self.activate_for_xy = nn.Sigmoid()
        self.activate_for_wh = nn.Softplus()
        self.activate_for_class = nn.Sigmoid()


    def forward(self, x):

        x = self.feature_layers(x)
        if torch.isnan(x.mean()):
            print(x)
            raise ValueError("1")
        x = self.yolo(x)
        x = x.reshape(-1, 7, 7, len(self.split_anchor_list[0]), 5+20)  # 重塑为标准输出格式
        x = x.reshape(-1, 7, 7, 3, 5+20)  # 重塑为标准输出格式

        # p，x，y, cls映射到0，1
        # x[:, :, :, :, 0] = (self.activate_for_confidence(x[:, :, :, :, 0])+1)/2
        if torch.isnan(x.mean()):
            print(x)
            raise ValueError("2")
        x[:, :, :, :, 1:3] = self.activate_for_xy(x[:, :, :, :, 1:3])
        if torch.isnan(x.mean()):
            print(x)
            raise ValueError("3")
        # x[:, :, :, :, 5:]=self.activate_for_class(x[:, :, :, :, 5:])
        # w，h映射到0，+INF
        x[:, :, :, :, 3:5] = (1+x[:, :, :, :, 3:5].exp()).log()
        if torch.isnan(x.mean()):
            print(x)
            raise ValueError("4")
        return [x]

# 返回符合条件的pxywhcls列表，暂时无法批量操作
    def forward_ret_pxywhcls_list(self, x0, nms_for_each_group=False,ret_origin_array=False):
        x = x0.transpose((2, 0, 1))
        x = torch.tensor(x, dtype=torch.float).to(device)
        if x.shape != (3, image_size, image_size):
            # x=F.pad()
            x = F.interpolate(x.unsqueeze(0), size=(image_size, image_size), mode="nearest")
        else:
            x=x.unsqueeze(0)
        output = self.forward(x)[0].squeeze(0).detach().cpu()  # shape(7,7,5,5+100)去掉batch
        output[...,0]=self.activate_for_confidence(output[...,0])
        output[...,5:]=self.activate_for_class(output[...,5:])
        grid_size=output.shape[0]

        output_list=[]
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(output.shape[2]):
                    if output[i,j,k,0]>=0.2:#置信度大于阈值时
                        #切片出当前数据并转换格式(相对于整体图像的归一化xywh)
                        temp=np.array(output[i,j,k])
                        temp[1]=(temp[1]+i)/grid_size
                        temp[2]=(temp[2]+j)/grid_size
                        temp[3]=temp[3]*self.split_anchor_list[0][k][0]/grid_size
                        temp[4]=temp[4]*self.split_anchor_list[0][k][1]/grid_size
                        class_id=np.argmax(temp[5:])
                        x1 = temp[1] - temp[3] / 2
                        y1 = temp[2] - temp[4] / 2
                        w = temp[3]
                        h = temp[4]
                        result=[temp[0],x1,y1,w,h,class_id]
                        if output[i,j,k,0]*temp[5:][class_id]>=0.005:
                            output_list.append(result)

        if output_list==[]:
            if ret_origin_array:
                return [[]], output.numpy()
            else:
                return [[]]


        # 创建一个字典，用于存储不同索引的分组
        groups = {}
        for sublist in output_list:
            # 找到子列表中最大元素的索引
            cls = sublist[-1]
            # 如果该索引不在字典中，则创建一个新的列表
            if cls not in groups:
                groups[cls] = []
            # 将子列表加入相应索引的分组中
            groups[cls].append(sublist)

        disposed_output_list=[]

        if nms_for_each_group:
            for element in groups:
                disposed_output_list.append(nms(groups[element]))
        else:
            disposed_output_list.append(nms(output_list))

        # 返回一个需要画出的框的列表[[[p,x,y,w,h,cls1],[p,x,y,w,h,cls1]],[[p,x,y,w,h,cls2],[p,x,y,w,h,cls2]]]
        if ret_origin_array:
            return disposed_output_list,output.numpy()
        else:
            return disposed_output_list

class BiggerMyNet(MyNet):
    def __init__(self):
        super().__init__()
        self.name="bigger_target_recog"
        self.layer1 = nn.Sequential(
            # input 416
            # nn.Conv2d(3, 64, 7, 2, 4),
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64,64,3,2,1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )
        self.layer2 = nn.Sequential(
            # 104
            MultiScaleConv(64, 64),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(),
            nn.Conv2d(192, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, 1, 1, 0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, 1, 1, 0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Dropout2d(0.15),
            #     接到yolo3
        )
        self.layer3 = nn.Sequential(
            # 52
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, 1, 1, 0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, 1, 1, 0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, 1, 1, 0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Dropout2d(0.15),
            #     接到up2
        )
        self.layer4 = nn.Sequential(
            # 26
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 256, 1, 1, 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 256, 1, 1, 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 256, 1, 1, 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Dropout2d(0.15),
            #     接到up1
        )
        self.layer5 = nn.Sequential(
            # 13
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 512, 1, 1, 0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 512, 1, 1, 0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 512, 1, 1, 0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 512, 1, 1, 0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 512, 1, 1, 0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Dropout2d(0.15),
        )

        # 每个yolo层使用的anchor不同
        self.yolo1 = nn.Sequential(
            # 最大的3个anchor
            nn.Conv2d(512, 75, 1, 1, 0),
            Reshape(-1, 13, 13, 3, 25),
            SelectiveSigmoid(1, 3),
            SelectiveExp(3, 5),
        )

        self.up1 = nn.Sequential(
            # 13
            nn.Conv2d(1024, 512, 1, 1, 0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Upsample((26, 26), mode="bilinear"),
            nn.Conv2d(512, 256, 1, 1, 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 256, 1, 1, 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 256, 1, 1, 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 256, 1, 1, 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 256, 1, 1, 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )

        self.yolo2 = nn.Sequential(
            nn.Conv2d(512, 75, 1, 1, 0),
            Reshape(-1, 26, 26, 3, 25),
            SelectiveSigmoid(1, 3),
            SelectiveExp(3, 5),
        )

        self.up2 = nn.Sequential(
            # 26
            nn.Conv2d(512, 256, 1, 1, 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Upsample((52, 52), mode="bilinear"),
            nn.Conv2d(256, 128, 1, 1, 0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, 1, 1, 0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, 1, 1, 0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, 1, 1, 0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, 1, 1, 0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, 1, 1, 0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, 1, 1, 0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )

        self.yolo3 = nn.Sequential(
            nn.Conv2d(256, 75, 1, 1, 0),
            Reshape(-1, 52, 52, 3, 25),
            SelectiveSigmoid(1, 3),
            SelectiveExp(3, 5),
        )




# net=BiggerMyNet()
# total_params = sum(p.numel() for p in net.parameters())
# print(f"Total number of parameters: {total_params}")