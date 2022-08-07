import numpy as np
import torch
from torch.nn import functional as F
from torchvision.ops import nms


def loc2bbox(src_bbox, loc):
    # 将边界框预测应用到生成的锚框上，也就是生成预测边界框
    # 预测边界框是由锚框和预测出的偏移量生成的
    # src_bbox就是每个锚框，内部数据为没个锚框的左上和右下坐标值  shape = (12996*9, 4)
    # data = [x1, y1, x2, y2]
    # loc是边界框预测, shape = (38*38*9, 4) = (12996*9, 4)
    if src_bbox.size()[0] == 0:
        return torch.zeros((0, 4), dtype=loc.dtype)

    src_width = torch.unsqueeze(src_bbox[:, 2] - src_bbox[:, 0], -1)
    # 得到每个锚框的W，unsqueeze之后 shape = (12996*9， 1）
    src_height = torch.unsqueeze(src_bbox[:, 3] - src_bbox[:, 1], -1)
    # 同理，获得每一个的H，shape = (12996*9, 1)
    src_ctr_x = torch.unsqueeze(src_bbox[:, 0], -1) + 0.5 * src_width
    src_ctr_y = torch.unsqueeze(src_bbox[:, 1], -1) + 0.5 * src_height   # shape = (12996*9, 1)
    # 得到中心点（锚点）坐标

    dx = loc[:, 0::4]  # 实际就是获得关于x1的预测值  shape = (12996*9,)
    dy = loc[:, 1::4]  # 同理就是关于y1预测值
    dw = loc[:, 2::4]  # x2
    dh = loc[:, 3::4]  # y2

    ctr_x = dx * src_width + src_ctr_x  # 中心点坐标加上偏移量
    ctr_y = dy * src_height + src_ctr_y
    w = torch.exp(dw) * src_width  # 偏移后的W
    h = torch.exp(dh) * src_height  # 偏移后的H
    # 计算方式实际就是李沐书上给的那种，只不过参数设置不一样，就是最常见的偏移量计算方式

    dst_bbox = torch.zeros_like(loc)  # 存储新的锚框信息
    dst_bbox[:, 0::4] = ctr_x - 0.5 * w
    dst_bbox[:, 1::4] = ctr_y - 0.5 * h
    dst_bbox[:, 2::4] = ctr_x + 0.5 * w
    dst_bbox[:, 3::4] = ctr_y + 0.5 * h  # 得到新的左上右下坐标信息

    return dst_bbox


class DecodeBox():
    def __init__(self, std, num_classes):
        self.std            = std
        self.num_classes    = num_classes + 1    

    def frcnn_correct_boxes(self, box_xy, box_wh, input_shape, image_shape):
        #-----------------------------------------------------------------#
        #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
        #-----------------------------------------------------------------#
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        box_mins    = box_yx - (box_hw / 2.)
        box_maxes   = box_yx + (box_hw / 2.)
        boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes

    def forward(self, roi_cls_locs, roi_scores, rois, image_shape, input_shape, nms_iou = 0.3, confidence = 0.5):
        results = []
        bs      = len(roi_cls_locs)
        #--------------------------------#
        #   batch_size, num_rois, 4
        #--------------------------------#
        rois    = rois.view((bs, -1, 4))
        #----------------------------------------------------------------------------------------------------------------#
        #   对每一张图片进行处理，由于在predict.py的时候，我们只输入一张图片，所以for i in range(len(mbox_loc))只进行一次
        #----------------------------------------------------------------------------------------------------------------#
        for i in range(bs):
            #----------------------------------------------------------#
            #   对回归参数进行reshape
            #----------------------------------------------------------#
            roi_cls_loc = roi_cls_locs[i] * self.std
            #----------------------------------------------------------#
            #   第一维度是建议框的数量，第二维度是每个种类
            #   第三维度是对应种类的调整参数
            #----------------------------------------------------------#
            roi_cls_loc = roi_cls_loc.view([-1, self.num_classes, 4])

            #-------------------------------------------------------------#
            #   利用classifier网络的预测结果对建议框进行调整获得预测框
            #   num_rois, 4 -> num_rois, 1, 4 -> num_rois, num_classes, 4
            #-------------------------------------------------------------#
            roi         = rois[i].view((-1, 1, 4)).expand_as(roi_cls_loc)
            cls_bbox    = loc2bbox(roi.contiguous().view((-1, 4)), roi_cls_loc.contiguous().view((-1, 4)))
            cls_bbox    = cls_bbox.view([-1, (self.num_classes), 4])
            #-------------------------------------------------------------#
            #   对预测框进行归一化，调整到0-1之间
            #-------------------------------------------------------------#
            cls_bbox[..., [0, 2]] = (cls_bbox[..., [0, 2]]) / input_shape[1]
            cls_bbox[..., [1, 3]] = (cls_bbox[..., [1, 3]]) / input_shape[0]

            roi_score   = roi_scores[i]
            prob        = F.softmax(roi_score, dim=-1)

            results.append([])
            for c in range(1, self.num_classes):
                #--------------------------------#
                #   取出属于该类的所有框的置信度
                #   判断是否大于门限
                #--------------------------------#
                c_confs     = prob[:, c]
                c_confs_m   = c_confs > confidence

                if len(c_confs[c_confs_m]) > 0:
                    #-----------------------------------------#
                    #   取出得分高于confidence的框
                    #-----------------------------------------#
                    boxes_to_process = cls_bbox[c_confs_m, c]
                    confs_to_process = c_confs[c_confs_m]

                    keep = nms(
                        boxes_to_process,
                        confs_to_process,
                        nms_iou
                    )
                    #-----------------------------------------#
                    #   取出在非极大抑制中效果较好的内容
                    #-----------------------------------------#
                    good_boxes  = boxes_to_process[keep]
                    confs       = confs_to_process[keep][:, None]
                    labels      = (c - 1) * torch.ones((len(keep), 1)).cuda() if confs.is_cuda else (c - 1) * torch.ones((len(keep), 1))
                    #-----------------------------------------#
                    #   将label、置信度、框的位置进行堆叠。
                    #-----------------------------------------#
                    c_pred      = torch.cat((good_boxes, confs, labels), dim=1).cpu().numpy()
                    # 添加进result里
                    results[-1].extend(c_pred)

            if len(results[-1]) > 0:
                results[-1] = np.array(results[-1])
                box_xy, box_wh = (results[-1][:, 0:2] + results[-1][:, 2:4])/2, results[-1][:, 2:4] - results[-1][:, 0:2]
                results[-1][:, :4] = self.frcnn_correct_boxes(box_xy, box_wh, input_shape, image_shape)

        return results
        
