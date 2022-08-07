# 创建ROI层以及之后的全连接分类网络，完成第二阶段工作
import warnings

import torch
from torch import nn
from torchvision.ops import RoIPool

warnings.filterwarnings("ignore")


class VGG16RoIHead(nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        super(VGG16RoIHead, self).__init__()
        self.classifier = classifier
        #--------------------------------------#
        #   对ROIPooling后的的结果进行回归预测
        #--------------------------------------#
        self.cls_loc    = nn.Linear(4096, n_class * 4)
        #-----------------------------------#
        #   对ROIPooling后的的结果进行分类
        #-----------------------------------#
        self.score      = nn.Linear(4096, n_class)
        #-----------------------------------#
        #   权值初始化
        #-----------------------------------#
        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.roi = RoIPool((roi_size, roi_size), spatial_scale)
        
    def forward(self, x, rois, roi_indices, img_size):
        n, _, _, _ = x.shape
        if x.is_cuda:
            roi_indices = roi_indices.cuda()
            rois = rois.cuda()
        rois        = torch.flatten(rois, 0, 1)
        roi_indices = torch.flatten(roi_indices, 0, 1)

        rois_feature_map = torch.zeros_like(rois)
        rois_feature_map[:, [0,2]] = rois[:, [0,2]] / img_size[1] * x.size()[3]
        rois_feature_map[:, [1,3]] = rois[:, [1,3]] / img_size[0] * x.size()[2]

        indices_and_rois = torch.cat([roi_indices[:, None], rois_feature_map], dim=1)
        #-----------------------------------#
        #   利用建议框对公用特征层进行截取
        #-----------------------------------#
        pool = self.roi(x, indices_and_rois)
        #-----------------------------------#
        #   利用classifier网络进行特征提取
        #-----------------------------------#
        pool = pool.view(pool.size(0), -1)
        #--------------------------------------------------------------#
        #   当输入为一张图片的时候，这里获得的f7的shape为[300, 4096]
        #--------------------------------------------------------------#
        fc7 = self.classifier(pool)

        roi_cls_locs    = self.cls_loc(fc7)
        roi_scores      = self.score(fc7)

        roi_cls_locs    = roi_cls_locs.view(n, -1, roi_cls_locs.size(1))
        roi_scores      = roi_scores.view(n, -1, roi_scores.size(1))
        return roi_cls_locs, roi_scores


class Resnet50RoIHead(nn.Module):
    # 第二阶段分类网络，不再是二元分类
    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        super(Resnet50RoIHead, self).__init__()
        self.classifier = classifier  # resnet50最后的部分

        #   ROI层之后的全连接网络
        self.cls_loc = nn.Linear(2048, n_class * 4)

        #   对ROIPooling后的的结果进行分类
        self.score = nn.Linear(2048, n_class)

        #   权值初始化
        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.roi = RoIPool((roi_size, roi_size), spatial_scale)
        # 使用官方roi层，spatial_scale是空间尺度
        # 按照文档说法，如果我的原图是224*224,经过卷积之后1feature map是112*112，说明下采样率是2
        # 如果我在原图取20*20的锚框，映射到feature map就是20/2=10，也就是spatial_scale设置为0.5=1/2

    def forward(self, x, rois, roi_indices, img_size):
        n, _, _, _ = x.shape   # batch-size

        if x.is_cuda:
            roi_indices = roi_indices.cuda()
            rois = rois.cuda()
        rois = torch.flatten(rois, 0, 1)
        # shape = (k, 4)  k表示保留下来的边界框数量，4表示x1 y1 x2 y2
        roi_indices = torch.flatten(roi_indices, 0, 1)  # shape = (k,)
        
        rois_feature_map = torch.zeros_like(rois)   # 在feature map上截取区域
        rois_feature_map[:, [0, 2]] = rois[:, [0, 2]] / img_size[1] * x.size()[3]
        # x是经过特征提取之后的feature map，记x.size()[3] = FW，而img_size是原图的尺寸，记img_size[1]=W
        # 这里感觉像是以原图大小W作为X轴上尺度，rois[:, [0, 2]] / img_size[1]获得比例，
        # 再乘上FW得到映射过去的X轴坐标
        rois_feature_map[:, [1, 3]] = rois[:, [1, 3]] / img_size[0] * x.size()[2]
        # 同理，获得Y轴映射

        indices_and_rois = torch.cat([roi_indices[:, None], rois_feature_map], dim=1)

        #   利用建议框对公用特征层进行截取
        pool = self.roi(x, indices_and_rois)

        #   resnet50的最后网络
        fc7 = self.classifier(pool)
        #--------------------------------------------------------------#
        #   当输入为一张图片的时候，这里获得的f7的shape为[300, 2048]
        # 但是，构建一个（1，3，600，600）的数据尝试，这里输出形状是（1，2048，2，2）
        #--------------------------------------------------------------#
        fc7 = fc7.view(fc7.size(0), -1)   # shape 应该变成（n, 2048*2*2)  n表示batch-size

        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)   # 两者同时进行，输入一致，就像是rpn的两个预测一样
        roi_cls_locs = roi_cls_locs.view(n, -1, roi_cls_locs.size(1))
        roi_scores = roi_scores.view(n, -1, roi_scores.size(1))
        return roi_cls_locs, roi_scores   # 返回候选区域分类损失以及最终边界框回归损失


def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
