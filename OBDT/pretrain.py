import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader,Subset
from torchvision import datasets, transforms
import os
from model import *
from datasets import CustomImageFolder
from config import *
from torch.utils.tensorboard import SummaryWriter
from information_printer import *


def evaluate_model(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        i=0
        for inputs, labels in val_loader:
            i+=1
            inputs, labels = inputs.to(torch.device("cuda")), labels.to(torch.device("cuda"))
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if i >=50:
                break

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the validation images: {accuracy}%')
    model.train()

def accuracy(output, target, topk=(1, 5)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.RandomAffine(
        degrees=30,
        translate=(0.1, 0.1),  # 水平和垂直方向的平移比例
        scale=(0.8, 1.2),  # 缩放比例范围
        shear=10  # 剪切角度范围
    ),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载数据集
train_dataset = CustomImageFolder(root="C:/Datasets/imagenet/ILSVRC2012_img_train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,num_workers=4)

val_dataset = datasets.ImageFolder(root="C:/Datasets/imagenet/ILSVRC2012_img_train", transform=val_transform)
# 定义子集索引 创建子集
subset_indices = torch.arange(0, 12000)  # 选择前x个样本
subset = Subset(val_dataset, subset_indices)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# 初始化模型、损失函数和优化器
model = Classifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# optimizer = optim.RMSprop(model.parameters(), lr=0.000001,momentum=0.85)
# optimizer = optim.SGD(model.parameters(), lr=0.000001,momentum=0.85)

# 定义学习率调度器
# scheduler = CyclicLR(optimizer, base_lr=0.000005, max_lr=0.00002, step_size_up=1000, mode='triangular',cycle_momentum=True)
# TODO:验证标签对应？初始化方法？
# 训练模型13000
num_epochs = 10000


writer = SummaryWriter(log_dir='logs/pretrain_classifier')
# 假设你有一个模型和输入
dummy_input = torch.randn(1, 3, 224, 224).to(device)  # 例如，对于一个输入形状为 [1, 3, 224, 224] 的模型
writer.add_graph(model, dummy_input)
if __name__ == '__main__':
    model.load_state_dict(torch.load(f"weights/pretrained/{model.name}_step_300000.pth"))

    step=0
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    evaluate_model(model,val_loader)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            # 将输入和标签转换为tensor
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)


            # 计算损失
            loss = criterion(outputs, labels)

            # 反向传播
            loss.backward()
            # print_grad(model)

            # 更新参数
            optimizer.step()

            # 梯度清零
            optimizer.zero_grad()
            # scheduler.step()

            # 记录损失
            running_loss += loss.item()
            step+=1


            if step%50==0:
                writer.add_scalar('loss',loss.item(),step)

            if step%500==0:
                print(f"step={step}, mean loss for 500 step={running_loss/500}")
                running_loss=0.0

            if step%10000==0:
                torch.save(model.state_dict(), f"weights/pretrained/{model.name}_step_{step}.pth")
                evaluate_model(model, val_loader)
                top1, top5 = accuracy(outputs, labels, topk=(1, 5))
                print(top1.item(),top5.item())
                print("model saved")

        print(f"Epoch [{epoch + 1}/{num_epochs}]")


    print('Finished。。怎么可能啊这么大数据集怎么可能啊这么大数据集怎么可能啊这么大数据集怎么可能啊这么大数据集绝对不可能')
