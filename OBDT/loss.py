import math
import torch.nn.functional as F
import torch
import torch.nn as nn

from utils import *
import config

# split_anchor_list=np.array([[[116,90],[156,198],[373,326]], [[30,61],[62,45],[59,119]], [[10,13],[16,30],[33,23]]],dtype=np.float32)
# grid=[13,26,52]
# for i in range(3):
#     split_anchor_list[i]=split_anchor_list[i]*grid[i]/416
# split_anchor_list=split_anchor_list.tolist()
all_anchor_list=[[3.625/13, 2.8125/13], [4.875/13, 6.1875/13], [11.65625/13, 10.1875/13], [1.875/26, 3.8125/26], [3.875/26, 2.8125/26], [3.6875/26, 7.4375/26], [1.25/52, 1.625/52], [2.0/52, 3.75/52], [4.125/52, 2.875/52]]


# 自定义损失函数，只针对错误分类样本计算损失
def selective_loss(logits, targets):
    logits = F.sigmoid(logits)
    # 计算交叉熵损失
    loss = F.cross_entropy(logits, targets, reduction='none')
    # 获取错误分类样本的掩码
    incorrect_mask = (logits.argmax(dim=1) != targets)
    # 只保留错误分类样本的损失
    selected_loss = loss[incorrect_mask]
    # 返回平均损失（可以根据需要调整为 sum 等）
    return selected_loss.mean() if len(selected_loss) > 0 else torch.tensor(0.0, requires_grad=True)

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        logits=F.sigmoid(logits)
        ce_loss = F.binary_cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # 预测正确概率
        focal_loss = (1 - pt) ** self.gamma * ce_loss  # 调整交叉熵损失
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def get_loss(predictions, targets, anchor_size_list,num=0):
    """

    :param predictions: 网络输出，batch*7*7*5*(p,x,y,w,h,cls1,cls2,cls3,...)
    :param targets: 读入的数据
    (tensor[[img,class,x,y,w,h],[img,class,x,y,w,h]],
     tensor[[img,class,x,y,w,h]],
     ...)
    :return: 标量张量与损失信息列表
    """
    # Check which device was used
    device="cuda" if torch.cuda.is_available() else "cpu"

    # Add placeholder variables for the different losses
    lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)


    # Build yolo labels
    labels = build_labels(predictions, targets, anchor_size_list)  # targets


    criterion_cls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))
    criterion_box = nn.MSELoss()
    # criterion_obj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([((num+1)*5)-3.7], device=device))
    # criterion_obj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([((num+1)*8)-6], device=device))
    criterion_obj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))
    # criterion_obj=selective_loss

    # 置信度损失
    predict_confidence,real_confidence=predictions[...,0],labels[...,0]
    lobj+=criterion_obj(predict_confidence,real_confidence)
    # lobj=criterion_obj(predict_confidence,real_confidence)
    # # 挑选损失前%k的输出进行回传
    # k=int(1*lobj.numel()/100)
    # # 先挑选出所有正类，它们都需要传播
    # condition_mask_p = real_confidence==1
    # condition_mask_n = real_confidence == 0
    # pos_obj=lobj[condition_mask_p]
    # neg_obj=lobj[condition_mask_n]
    #
    # # 在把正类滤除后，挑选负类中前%k，与所有正类concat并求其mean作为损失值
    # top_values, top_indices = torch.topk(neg_obj, k)
    # lobj=torch.cat([pos_obj,top_values]).mean()



    #TODO:对目标置信度为1的网格单元中的
    # 5个预设中与目标框的iou最大的那个预设计算定位与类别损失

    condition_mask=labels[...,0]!=0
    predictions=predictions[condition_mask]
    labels=labels[condition_mask]

    # 对目标置信度为1的网格单元计算类别损失
    predict_cls,real_cls=predictions[...,5:],labels[...,5:]
    lcls+=criterion_cls(predict_cls,real_cls)

    # 对目标置信度为1的网格单元计算定位损失
    pred_xywh=torch.zeros_like(predictions[...,1:5])
    real_xywh=torch.zeros_like(labels[...,1:5])
    pred_xywh[...,0:2]=predictions[...,1:3]
    pred_xywh[...,2:4]=predictions[...,3:5]**0.5
    real_xywh[..., 0:2] = labels[..., 1:3]
    real_xywh[..., 2:4] = labels[...,3:5]**0.5


    # pred_xywh=xywh2xyxy(pred_xywh)
    # real_xywh=xywh2xyxy(real_xywh)
    lbox+=criterion_box(pred_xywh,real_xywh)





    # iou=bbox_iou(pred_xywh,real_xywh)
    # mean_iou=iou.mean()
    # lbox+=(1-mean_iou)

    # TODO:算法略有不同，调节这几个权重，box和cls大一些,lbox值溢出问题
    # Merge losses
    if torch.isnan(lbox):
        lbox=torch.zeros(1).to(device)
    if torch.isnan(lcls):
        lcls=torch.zeros(1).to(device)

    lbox *= 0.2
    lobj *= 1.0
    lcls *= 0.1
    loss = lbox + lobj + lcls

    return loss, torch.cat((lbox, lobj, lcls, loss)).detach()

def build_labels(predictions, targets, anchor_size_list):
    # x,y,w,h是模型标准输出格式的
    number_of_anchor=len(anchor_size_list)

    grid_size=predictions.shape[1]
    # 标签与预测输出形状一致
    # labels=torch.zeros_like(predictions)#batch*7*7*5*(p,x,y,w,h,cls)
    labels =predictions.clone().detach() # batch*7*7*5*(p,x,y,w,h,cls)
    # 标签中置信度的所有值置0
    labels[...,0]=0
    # 标签中类别独热编码的所有值置0
    labels[..., 5:] = 0
    # labels[...,-1]=1


    # 逐张取批次中的N张图，构建N个标签
    for i,target_per_img in enumerate(targets):
        target_per_img=target_per_img.clone()
        for tar in target_per_img:#tar:[img,class,x,y,w,h]
            if torch.isnan(tar.mean()):
                print(tar)
                raise ValueError("tar nan")

            # 框坐标格式转换
            tar[2:]=xywh_drawbox2xywh_label(tar[2:])
            # 取出cls,x,y,w,h
            # 0维是图片id，从1维开始取
            cls,x,y,w,h=tar[...,1],tar[...,2],tar[...,3],tar[...,4],tar[...,5]
            if x>1:print("X",x)
            if y > 1: print("Y",y)
            #独热编码
            one_hot_code=one_hot_encoding(int(cls))
            # 判断在哪个网格里，计算相对坐标
            grid_x, grid_y, relative_x, relative_y = get_grid_cell_with_normalized_coords(x, y, grid_size=grid_size)
            # 给该网格内的所有anchor填充标签
            #TODO:如果第j个框是iou最大的，就填充真实坐标与类别，否则坐标不做任何填充 #类别填背景类？

            iou_between_anchor_and_target=[]
            current_grid=labels[i][grid_x][grid_y]
            for j in range(number_of_anchor):
                iou_between_anchor_and_target.append(bbox_wh_iou(torch.tensor([anchor_size_list[j][0]/grid_size,anchor_size_list[j][1]/grid_size]),torch.tensor([w,h])))
            # 取与目标iou最大的框
            j=iou_between_anchor_and_target.index(max(iou_between_anchor_and_target))

            iou_for_all_anchors=[]
            for x in range(len(all_anchor_list)):
                iou_for_all_anchors.append(bbox_wh_iou(torch.tensor([all_anchor_list[x][0],all_anchor_list[x][1]]),torch.tensor([w,h])))

            if 1:
            # if iou_between_anchor_and_target[j] == max(iou_for_all_anchors):
                # 如果目标大小属于当前的检测层职能范围就填充
                anchor_w=anchor_size_list[j][0]
                anchor_h=anchor_size_list[j][1]
                current_grid[j][0]=1#TODO:最后1改成iou值试试
                current_grid[j][1]=relative_x
                current_grid[j][2]=relative_y
                current_grid[j][3]=w*grid_size/anchor_w
                current_grid[j][4]=h*grid_size/anchor_h
                current_grid[j][5:]=one_hot_code


    return labels


# 测试用函数@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def tbuild_labels(predictions, targets, anchor_size_list):
    # x,y,w,h是模型标准输出格式的
    number_of_anchor=len(anchor_size_list)

    grid_size=predictions.shape[1]
    # 标签与预测输出形状一致
    # labels=torch.zeros_like(predictions)#batch*7*7*5*(p,x,y,w,h,cls)
    labels =predictions.clone().detach() # batch*7*7*5*(p,x,y,w,h,cls)
    # 标签中置信度的所有值置0
    labels[...,0]=0
    # 标签中类别独热编码的所有值置0
    labels[..., 5:] = 0
    # labels[...,-1]=1


    # 逐张取批次中的N张图，构建N个标签
    for i,target_per_img in enumerate(targets):
        target_per_img=target_per_img.clone()
        for tar in target_per_img:#tar:[img,class,x,y,w,h]
            # 框坐标格式转换
            tar[2:]=xywh_drawbox2xywh_label(tar[2:])
            # 取出cls,x,y,w,h
            # 0维是图片id，从1维开始取
            cls,x,y,w,h=10,0.2307,0.5,0.2307,0.2307
            if x>1:print("X",x)
            if y > 1: print("Y",y)
            #独热编码
            one_hot_code=one_hot_encoding(10)
            # 判断在哪个网格里，计算相对坐标
            grid_x, grid_y, relative_x, relative_y = get_grid_cell_with_normalized_coords(x, y, grid_size=grid_size)
            # 给该网格内的所有anchor填充标签
            #TODO:如果第j个框是iou最大的，就填充真实坐标与类别，否则坐标不做任何填充 #类别填背景类？

            iou_between_anchor_and_target=[]
            current_grid=labels[i][grid_x][grid_y]
            for j in range(number_of_anchor):
                iou_between_anchor_and_target.append(bbox_wh_iou(torch.tensor([anchor_size_list[j][0]/grid_size,anchor_size_list[j][1]/grid_size]),torch.tensor([w,h])))

            # 取与目标iou最大的框
            j=iou_between_anchor_and_target.index(max(iou_between_anchor_and_target))
            anchor_w=anchor_size_list[j][0]
            anchor_h=anchor_size_list[j][1]
            current_grid[j][0]=1#TODO:最后1改成iou值试试
            current_grid[j][1]=relative_x
            current_grid[j][2]=relative_y
            current_grid[j][3]=w*grid_size/anchor_w
            current_grid[j][4]=h*grid_size/anchor_h
            current_grid[j][5:]=one_hot_code

    return labels