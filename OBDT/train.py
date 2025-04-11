import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import argparse

import numpy as np
import tqdm

import torch
from torch.utils.data import DataLoader,Subset
from torch.optim.lr_scheduler import CyclicLR, StepLR
from pycocotools.coco import COCO
# from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CocoDetection
from torchvision import transforms

from datasets import *
from model import MyNet,MiniDetector,ModifiedResnet,BiggerMyNet
from loss import get_loss
from utils import concatenate_tensors_with_batch, compute_precise_for_images,check_grad,compute_precise
from config import *

from terminaltables import AsciiTable

from torchsummary import summary

# 网络及优化器
net = MyNet().to(device)
# net = BiggerMyNet().to(device)
# net=MiniDetector().to(device)
# net=ModifiedResnet().to(device)
# net.freeze()

# optimizer = torch.optim.RMSprop(net.parameters(), lr=lr/10, momentum=0.8)
# optimizer = torch.optim.SGD(net.parameters(), lr=10*lr, momentum=0.9, weight_decay=0.0005)
optimizer = torch.optim.AdamW(net.parameters(), lr=lr,weight_decay=0.0005)


# 定义学习率调度器
# scheduler = CyclicLR(optimizer, base_lr=lr/5, max_lr=lr, step_size_up=500, mode='triangular',cycle_momentum=False)
scheduler = StepLR(optimizer,1000,0.95)


# 定义数据集的路径
path_to_train_image_folder = "C:/Datasets/VOC_2012"
path_to_train_annotation_file = "C:/Datasets/coco_2014/annotations】】/instances_train2014.json"

path_to_val_image_folder = "C:/Datasets/VOC_2012"
path_to_val_annotation_file = "C:/Datasets/coco_2014/annotations/instances_val2014.json"

# 创建 Data
voc_dataset_train = CustomVOCDetection(root=path_to_train_image_folder,image_set="trainval",
                                        transform=AUGMENTATION_TRANSFORMS if augment else DEBUG_TRANSFORMS)
voc_dataset_val=CustomVOCDetection(root=path_to_val_image_folder,image_set="val",
                                        transform=DEBUG_TRANSFORMS)

# 定义子集索引 创建子集
subset_indices = torch.arange(0, subset_size)  # 选择前x个样本
subset = Subset(voc_dataset_train, subset_indices)
train_data_loader = DataLoader(subset if use_subset else voc_dataset_train, batch_size=batch_size, shuffle=shuffle, collate_fn=voc_dataset_train.collate_fn)
val_data_loader=DataLoader(Subset(voc_dataset_val, torch.arange(0, 1024)), batch_size=64, shuffle=False, collate_fn=voc_dataset_val.collate_fn,num_workers=4)

# writer = SummaryWriter(log_dir=f'logs/{net.name}')

# TODO:在梯度裁剪关闭时寻找不会梯度爆炸的最佳学习率，乘以0.1
if __name__ == '__main__':
    # 2592000
    # te.load_state_dict(torch.load("weights/pretrained/target_recognition_pretrained_step_710000.pth"))
    # net.feature_layers.requires_grad_(False)
    # net.feature_layers.load_state_dict(te.feature_layers.state_dict())
    net.load_state_dict(torch.load("weights/target_recognition_step_2600000.pth"))

    # net.load_state_dict(torch.load("weights/bigger_target_recog_step_508000.pth",weights_only=True))
    # net.load_state_dict(torch.load("weights/res_yolo_step_1000.pth"))
    step = 2600000
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total number of parameters: {total_params}")
    running_lbox, running_lobj, running_lcls, running_loss = 0.0, 0.0, 0.0, 0.0
    for epoch in range(epochs):
        net.train()
        print(f"epoch={epoch}")
        for batch_i, (imgs, targets) in enumerate(train_data_loader):

            if step % step_evaluate == 0:
                net.eval()
                print("train data:*************************************************")
                compute_precise_for_images(imgs, targets, net)#训练集当前batch测试准确率
                print("valid data:*************************************************")
                # compute_precise(net,val_data_loader)#验证集测试准确率

                # 测试验证机损失aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
                # with torch.no_grad():
                #     running_lboxx =0
                #     running_lobjx =0
                #     running_lclsx =0
                #     running_lossx =0
                #     lossx = torch.zeros(1).to(device)
                #     for batch_q, (imgsx, targetsx) in enumerate(val_data_loader):
                #         tensor_for_input = concatenate_tensors_with_batch(imgsx).to(device)
                #         output = net(tensor_for_input)
                #
                #         # 给每个yolo预测层都计算一遍损失
                #         for i in range(len(output)):
                #             lsx, loss_info_list = get_loss(output[i], targetsx, net.split_anchor_list[i])
                #             lossx += lsx
                #             running_lboxx += loss_info_list[0]
                #             running_lobjx += loss_info_list[1]
                #             running_lclsx += loss_info_list[2]
                #             running_lossx += loss_info_list[3]
                #     print(f"loss in val: lbox:{running_lboxx/len(val_data_loader)},lobj:{running_lobjx/len(val_data_loader)},lcls:{running_lclsx/len(val_data_loader)}")
                # 测试验证机损失aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa


                net.train()

            tensor_for_input = concatenate_tensors_with_batch(imgs).to(device)
            output = net(tensor_for_input)

            loss=torch.zeros(1).to(device)
            # 给每个yolo预测层都计算一遍损失
            for i in range(len(output)):
                ls, loss_info_list = get_loss(output[i], targets,net.split_anchor_list[i],num=i)
                loss+=ls
                running_lbox += loss_info_list[0]
                running_lobj += loss_info_list[1]
                running_lcls += loss_info_list[2]
                running_loss += loss_info_list[3]
            loss.backward()


            # check_grad(net)
            optimizer.step()
            optimizer.zero_grad()
            if use_scheduler:
                scheduler.step()




            step += 1
            if step%50==0:
                pass
                # writer.add_scalar('loss_box',loss_info_list[0], step)
                # writer.add_scalar('loss_obj', loss_info_list[1], step)
                # writer.add_scalar('loss_cls', loss_info_list[2], step)
                # writer.add_scalar('loss', loss_info_list[3], step)
            if step % step_save == 0:
                torch.save(net.state_dict(), f"weights/{net.name}_step_{step}.pth")
                print("model saved")

            if step % step_print_loss == 0:
                print(f"step={step}")
                print(f"lr={scheduler.get_last_lr()}")
                print(
                    f"mean loss for {step_print_loss} steps: lbox={running_lbox / step_print_loss},lobj={running_lobj / step_print_loss},lcls={running_lcls / step_print_loss},total loss={running_loss / step_print_loss}")
                running_lbox, running_lobj, running_lcls, running_loss = 0.0, 0.0, 0.0, 0.0


