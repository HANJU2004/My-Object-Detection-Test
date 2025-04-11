import torch
from torch import nn


class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConv, self).__init__()
        # 3x3, 5x5, 7x7卷积
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv7x7 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3)

        # 批归一化
        self.bn = nn.BatchNorm2d(out_channels * 3)

    def forward(self, x):
        # 通过不同的卷积层
        out_3x3 = self.conv3x3(x)
        out_5x5 = self.conv5x5(x)
        out_7x7 = self.conv7x7(x)

        # 将输出在通道维度上拼接
        out = torch.cat([out_3x3, out_5x5, out_7x7], dim=1)

        # 进行批归一化
        out = self.bn(out)
        return out


class SelectiveSigmoid(nn.Module):
    def __init__(self,start,end):
        super(SelectiveSigmoid, self).__init__()
        self.sigmoid = nn.Tanh()
        self.s=start
        self.e=end

    def forward(self, x):
        # 选择特定通道 (例如，只对第2通道及其后的通道激活)
        x[..., self.s:self.e] = (1+self.sigmoid(x[..., self.s:self.e]))/2
        return x

class SelectiveSoftplus(nn.Module):
    def __init__(self,start,end):
        super(SelectiveSoftplus, self).__init__()

        self.s=start
        self.e=end

    def forward(self, x):
        # 选择特定通道 (例如，只对第2通道及其后的通道激活)
        x[..., self.s:self.e] = (x[..., self.s:self.e].exp()+1).log()
        return x

class SelectiveExp(nn.Module):
    def __init__(self,start,end):
        super(SelectiveExp, self).__init__()

        self.s=start
        self.e=end

    def forward(self, x):
        # 选择特定通道 (例如，只对第2通道及其后的通道激活)
        x[..., self.s:self.e] = x[..., self.s:self.e].exp()
        return x

class Reshape(nn.Module):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(*self.shape)  # 保持批次维度不变