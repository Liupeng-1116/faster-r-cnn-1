# 定义区域建议网络 RPN
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import nms
from utils.anchors import _enumerate_shifted_anchor, generate_anchor_base
from utils.utils_bbox import loc2bbox


class ProposalCreator(object):
    def __init__(
        self, 
        mode, 
        nms_iou=0.7,  # NMS阈值
        n_train_pre_nms=12000,
        n_train_post_nms=600,
        n_test_pre_nms=3000,
        n_test_post_nms=300,
        min_size=16
    ):
        #   设置处于预测或者训练模式
        self.mode = mode

        #   NMS阈值
        self.nms_iou = nms_iou

        #   训练用到的建议框数量
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms

        #   预测用到的建议框数量
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms

        self.min_size = min_size

    def __call__(self, loc, score, anchor, img_size, scale=1.):
        # 边界框预测， 二元分类， 锚框， 图像尺寸，缩放比

        if self.mode == "training":  # 训练模式
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:  # 验证模式
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        #   anchor转换为tensor
        anchor = torch.from_numpy(anchor).type_as(loc)  # 设置元素数据类型与loc相同

        #   将RPN网络预测结果转化成建议框（原锚框加上偏移值）
        roi = loc2bbox(anchor, loc)  # 新的锚框坐标信息，[x1, y1, x2, y2]
        # shape = (12996, 4)

        #   防止建议框超出图像边缘
        roi[:, [0, 2]] = torch.clamp(roi[:, [0, 2]], min=0, max=img_size[1])
        # 限制roi[0, 2]的值都在 0 ~~ max之间
        # 实际上，max= img_size[1]就是输入的宽度W，就是这张图片在X轴上最大的x值。
        # 也就是限制左上和右下的坐标x1 x2都不能超过这个最大x值
        roi[:, [1, 3]] = torch.clamp(roi[:, [1, 3]], min=0, max=img_size[0])
        # 同理，限制y值

        #   设置建议框的宽高的最小值不可以小于16
        min_size = self.min_size * scale
        keep = torch.where(((roi[:, 2] - roi[:, 0]) >= min_size) & ((roi[:, 3] - roi[:, 1]) >= min_size))[0]
        roi = roi[keep, :]   # 将宽高达标的保留下来
        score = score[keep]

        #   根据置信度进行排序，取出一定量满足条件的建议框
        order = torch.argsort(score, descending=True)  # 降序排序
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        roi = roi[order, :]
        score = score[order]  # 根据置信度取前n_pre_nms名

        #   对建议框进行非极大抑制
        #   使用官方的非极大抑制会快非常多，注意它的返回值实际上是保留下建议框的索引，并且是按score降序排序
        keep = nms(roi, score, self.nms_iou)

        if len(keep) < n_post_nms:
            # 如果经过NMS后剩下的建议框数量不足够设置的n_post_nms这个量，
            # 就随机生成一些整数作为补充索引
            index_extra = np.random.choice(range(len(keep)), size=(n_post_nms - len(keep)), replace=True)
            keep = torch.cat([keep, keep[index_extra]])   # 连接补充索引
        keep = keep[:n_post_nms]   # 根据NMS结果取前n_post_nms个
        roi = roi[keep]
        # 一共经过两次过滤建议框，一次筛掉score低的，
        # 第二次筛掉IOU高的，并且保留也是按照置信度高低进行保留，最终保留的是置信度高而且经过NMS
        return roi


class RegionProposalNetwork(nn.Module):
    # 区域建议网络
    # 按照之前设定的假设输入图像都裁剪成为（3，600，600），
    # 那么特征提取网络送过来的feature map形状为（1024，38，38）
    def __init__(
        self, 
        in_channels=512,
        mid_channels=512,
        ratios=[0.5, 1, 2],   # 纵横比（宽高比）
        anchor_scales=[8, 16, 32],  # 缩放比
        feat_stride=16,
        mode="training",
    ):
        super(RegionProposalNetwork, self).__init__()

        #   生成基础锚框，shape为[9, 4]
        self.anchor_base = generate_anchor_base(anchor_scales=anchor_scales, ratios=ratios)
        # shape = (9, 4)
        # 返回的是每个锚框的半高宽  一共9个，每个都是[-H/2, -W/2, H/2, W/2]
        n_anchor = self.anchor_base.shape[0]   # 9

        #   对feature map 先进行一个3x3的卷积，可理解为特征整合
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)

        #   进行二元类别预测，检测锚框内部是否含有物体
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        # 输出有18个通道, (18,38,38)

        #   边界框预测（与二元类别预测同时进行，输入相同）
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)
        # 输出有36个通道, (36,38,38)

        #   特征点间距步长
        self.feat_stride = feat_stride

        #   用于对建议框解码并进行非极大抑制
        self.proposal_layer = ProposalCreator(mode)

        #   对RPN的网络部分进行权值初始化
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_size, scale=1.):
        n, _, h, w = x.shape   # n表示batch-size

        #   先进行一个3x3的卷积，可理解为特征整合
        x = F.relu(self.conv1(x))

        #   回归预测对先验框进行调整
        rpn_locs = self.loc(x)   # shape = (n, 36, 38, 38)
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        # (n, 38*38*9, 4) 表示每章图像一共有38*38*9个锚框，每个锚框4个预测坐标值（或者说是坐标值的预测偏移）

        #   分类预测先验框内部是否包含物体
        rpn_scores = self.score(x)  # (n, 18,38,38)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous().view(n, -1, 2)
        # (n, 38*38*9, 2) 同理，只不过这里就成了二元分类
        # 最后的2表示 [x, y] x表示不含物体，y表示含物体

        """感觉这里计算出的，是被拿去作为边界框的置信度了，最终RPN网络并不返回下面的
        而且在训练时，会拿rpn_scores计算损失函数值，所以应该就是拿下面这个作为置信度了"""
        #   进行softmax概率计算，每个先验框只有两个判别结果（二元分类）
        #   内部包含物体或者内部不包含物体，rpn_softmax_scores[:, :, 1]的内容为包含物体的概率
        rpn_softmax_scores = F.softmax(rpn_scores, dim=-1)
        # 对(n,38*38*9,2)的最后一维应用softmax函数
        rpn_fg_scores = rpn_softmax_scores[:, :, 1].contiguous()
        # shape = (n*38*38*9,)，只取最后一列，也就是含物体的预测概率结果
        rpn_fg_scores = rpn_fg_scores.view(n, -1)  # shape = (n, 38*38*9)
        # 表示对每张图片中所有锚框的二元分类预测含物体结果

        #  生成锚框，此时获得的anchor是布满网格点的，当输入图片为600,600,3的时候，shape为(12996, 4)
        # 之前已经获得了每个锚框的半高宽，得到锚点坐标，就可以生成锚框
        anchor = _enumerate_shifted_anchor(np.array(self.anchor_base), self.feat_stride, h, w)
        # shape = (12996 , 4)

        rois = list()
        roi_indices = list()
        for i in range(n):  # 遍历batch中每一张图片
            roi = self.proposal_layer(rpn_locs[i], rpn_fg_scores[i], anchor, img_size, scale=scale)
            batch_index = i * torch.ones((len(roi),))
            rois.append(roi.unsqueeze(0))
            roi_indices.append(batch_index.unsqueeze(0))

        rois = torch.cat(rois, dim=0).type_as(x)  # 最终给出的建议框
        roi_indices = torch.cat(roi_indices, dim=0).type_as(x)  # 建议框索引
        anchor = torch.from_numpy(anchor).unsqueeze(0).float().to(x.device)
        
        return rpn_locs, rpn_scores, rois, roi_indices, anchor
    # 返回rpn边界框预测、二元分类结果、预测边界框、预测边界框索引、锚框


def normal_init(m, mean, stddev, truncated=False):
    # 正态分布初始化
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
