from __future__ import division

import time
import platform
import tqdm
import torch
import torch.nn as nn
import torchvision
import numpy as np
import subprocess
import random
import imgaug as ia

# 所有的数据变换模块都写在这里，格式转换，坐标变换等

# 让我看看有没有梯度消失
def check_grad(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"Parameter: {name}, Gradient mean: {param.grad.mean()}")



# 权重初始化
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)





def get_grid_cell_with_normalized_coords(normalized_x, normalized_y, grid_size=7):
    """
    根据输入的归一化坐标判断该点在图像的哪个网格单元内，
    并返回该点相对该网格单元内的归一化坐标。

    :param normalized_x: 点的归一化x坐标，范围为[0, 1]
    :param normalized_y: 点的归一化y坐标，范围为[0, 1]
    :param grid_size: 网格的尺寸，默认是7
    :return: (grid_x, grid_y, cell_x, cell_y) 网格单元的索引及该点在网格单元内的归一化坐标
    """
    if not (0 <= normalized_x <= 1) or not (0 <= normalized_y <= 1):
        raise ValueError("Normalized coordinates should be within the range [0, 1].")

    grid_x = int(normalized_x * grid_size)
    grid_y = int(normalized_y * grid_size)

    # 边界条件处理，确保索引在网格范围内
    if grid_x == grid_size:
        grid_x -= 1
    if grid_y == grid_size:
        grid_y -= 1

    # 计算点在所属网格单元内的归一化坐标
    cell_x = (normalized_x * grid_size) - grid_x
    cell_y = (normalized_y * grid_size) - grid_y

    return grid_x, grid_y, cell_x, cell_y


def one_hot_encoding(integer, dimension=20):
    one_hot = torch.zeros(dimension)
    one_hot[integer-1] = 1
    return one_hot



# 用于模型输出计算iou
def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


# 用作从coco读取数据时，转换为对角格式输入增强模块
def xywh2xyxy_np(x):
    y = np.zeros_like(x)
    y[..., 0] = x[..., 0]# - x[..., 2] / 2
    y[..., 1] = x[..., 1]# - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2]
    y[..., 3] = x[..., 1] + x[..., 3]
    return y

# 用作模型输出时，转换为coco/plt格式用作绘制边框
def xywh2xywh_drawbox(x):
    y = torch.zeros_like(x)

    pass


# 用作制作标签时，转换为模型读取的中心xywh格式
def xywh_drawbox2xywh_label(x):
    y = torch.zeros_like(x)
    y[..., 0] = x[..., 0] + x[..., 2] / 2
    y[..., 1] = x[..., 1] + x[..., 3] / 2
    y[..., 2] = x[..., 2]
    y[..., 3] = x[..., 3]
    return y


def concatenate_tensors_with_batch(tensor_tuple):
    """
    将元组内的tensor组成一个batch

    Parameters:
    tensor_tuple (tuple): A tuple of tensors to concatenate.

    Returns:
    torch.Tensor: The concatenated tensor with an added batch dimension.
    """
    # Add a new dimension to each tensor
    tensors_with_batch_dim = [t.unsqueeze(0) for t in tensor_tuple]
    # Concatenate along the new batch dimension
    concatenated_tensor = torch.cat(tensors_with_batch_dim, dim=0)
    return concatenated_tensor



def compute_precise_for_images(img_list_batch, target_list_batch, model,dataloader=None):
    '''
    计算定位精度，对于1批次图片，给定网络输出（list）与实际框（list），暂时不考虑分类是否正确
    对于每个实际框，遍历输出结果，如果存在一个输出与该实际框的iou大于0.60，记为识别正确，真正类计数加一，跳转至下一个实际框。
    精度为：真正类数/len（实际框）  召回率为：真正类数/len（输出）
    '''
    result=0
    ra=0
    rb=0
    for i in range(len(img_list_batch)):
        tp=0
        img_for_input=img_list_batch[i].numpy().transpose((1,2,0))
        output_list=model.forward_ret_pxywhcls_list(img_for_input,nms_for_each_group=False)
        num_output=len(output_list[0])
        if num_output==0:
            continue
        num_target=len(target_list_batch[i])
        for target in target_list_batch[i]:
            for output in output_list[0]:
                target_bbox=target[2:]
                output_bbox=torch.tensor(output[1:5])
                if bbox_iou(output_bbox,target_bbox,x1y1x2y2=False)>=0.50:
                    tp+=1
                    break
        a=tp/num_output
        b=tp/num_target
        ra+=a
        rb+=b
        result+=(a+b)/2

    print(f"tp/num_output= {ra/len(img_list_batch)}, tp/num_target= {rb/len(img_list_batch)}")
    print(f"average precise for {len(img_list_batch)} images: {result/len(img_list_batch)}")
    return result/len(img_list_batch)

@torch.no_grad()
def compute_precise(model,dataloader):
    '''
    计算定位精度，对于1批次图片，给定网络输出（list）与实际框（list），暂时不考虑分类是否正确
    对于每个实际框，遍历输出结果，如果存在一个输出与该实际框的iou大于0.60，记为识别正确，真正类计数加一，跳转至下一个实际框。
    精度为：真正类数/len（实际框）  召回率为：真正类数/len（输出），二者求平均即为不考虑分类的定位精度
    '''
    result=0
    ra=0
    rb=0
    for img_list_batch,target_list_batch in dataloader:
        for i in range(len(img_list_batch)):
            tp=0
            img_for_input=img_list_batch[i].numpy().transpose((1,2,0))
            output_list=model.forward_ret_pxywhcls_list(img_for_input,nms_for_each_group=False)
            num_output=len(output_list[0])
            if num_output==0:
                continue
            num_target=len(target_list_batch[i])
            for target in target_list_batch[i]:
                for output in output_list[0]:
                    target_bbox=target[2:]
                    output_bbox=torch.tensor(output[1:5])
                    if bbox_iou(output_bbox,target_bbox,x1y1x2y2=False)>=0.50:
                        tp+=1
                        break
            a=tp/num_output
            b=tp/num_target
            ra+=a
            rb+=b
            result+=(a+b)/2

    print(f"tp/num_output= {(ra/len(img_list_batch))/len(dataloader)}, tp/num_target= {(rb/len(img_list_batch))/len(dataloader)}")
    print(f"average precise for {len(img_list_batch)*len(dataloader)} images: {(result/len(img_list_batch))/len(dataloader)}")
    return result/len(img_list_batch)




# 用来计算anchor box与目标的iou用于填充
def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[..., 0] , box1[..., 0] + box1[..., 2]
        b1_y1, b1_y2 = box1[..., 1] , box1[..., 1] + box1[..., 3]
        b2_x1, b2_x2 = box2[..., 0] , box2[..., 0] + box2[..., 2]
        b2_y1, b2_y2 = box2[..., 1] , box2[..., 1] + box2[..., 3]
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = \
            box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = \
            box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1e-10, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1e-10, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1e-10) * (b1_y2 - b1_y1 + 1e-10)
    b2_area = (b2_x2 - b2_x1 + 1e-10) * (b2_y2 - b2_y1 + 1e-10)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-10)

    return iou


def nms(bounding_boxes, threshold=0.5):
    """
        Non-max Suppression Algorithm

        @param list  Object candidate bounding boxes[[p,x,y,w,h,cls],[p,x,y,w,h,cls],...]
        @param list  Confidence score of bounding boxes
        @param float IoU threshold

        @return Rest boxes after nms operation
    """
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    score=boxes[:,0]
    # coordinates of bounding boxes
    start_x = boxes[:, 1]
    start_y = boxes[:, 2]
    end_x = boxes[:, 3]+start_x
    end_y = boxes[:, 4]+start_y

    # Confidence scores of bounding boxes
    # score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 0.01) * (end_y - start_y + 0.01)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(score[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 0.01)
        h = np.maximum(0.0, y2 - y1 + 0.01)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return picked_boxes#, picked_score





# TODO:计算准确度，仅测试时使用
def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")

# TODO:
def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap