# 关于锚框的函数，生成锚框半高宽、生成所有锚框
import numpy as np

#--------------------------------------------#
#   生成基础的先验框
#--------------------------------------------#


def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4), dtype=np.float32)
    # 每个锚点要有9个锚框，每个锚框用4个坐标点描述，所以是（9，4）
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(anchor_scales) + j   # 锚框索引
            anchor_base[index, 0] = - h / 2.
            anchor_base[index, 1] = - w / 2.
            anchor_base[index, 2] = h / 2.
            anchor_base[index, 3] = w / 2.
            # 获得锚框半高宽
    return anchor_base

#--------------------------------------------#
#   对基础先验框进行拓展对应到所有特征点上（就是生成所有锚框）
#--------------------------------------------#


def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    #   计算网格中心点
    shift_x = np.arange(0, width * feat_stride, feat_stride)   # shape = (38,)
    # 在x轴上均匀取点，但是并没有归一化
    shift_y = np.arange(0, height * feat_stride, feat_stride)  # shape = (38,)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)  # shape = (38,38)
    # 生成网格点x、y坐标矩阵
    shift = np.stack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel(),), axis=1)
    # ravel() 函数，作用类似flatten和reshape以及squeeze函数，ravel直接将多维数组展平成1维
    # 此时是（X,Y,X,Y)  shape = (38*38, 4) 表示一共有38*38个锚点，每个锚框四个坐标值

    #   每个网格点上的9个先验框
    A = anchor_base.shape[0]  # 9
    K = shift.shape[0]  # 38*38
    anchor = anchor_base.reshape((1, A, 4)) + shift.reshape((K, 1, 4))  # shape = (38*38, 9, 4)
    # 借助numpy广播特性，实现（X-W1/2,Y-H1/2,X+W1/2,Y+H1/2)， 也就是得到所有锚框左上和右下的坐标值

    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    # shape = (38*38*9, 4) = (12996, 4)  表示所有锚框的左上和右下坐标值
    return anchor


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    nine_anchors = generate_anchor_base()
    print(nine_anchors)

    height, width, feat_stride = 38, 38, 16
    anchors_all = _enumerate_shifted_anchor(nine_anchors, feat_stride, height, width)
    print(np.shape(anchors_all))
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ylim(-300, 900)
    plt.xlim(-300, 900)
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    plt.scatter(shift_x,shift_y)
    box_widths  = anchors_all[:,2]-anchors_all[:,0]
    box_heights = anchors_all[:,3]-anchors_all[:,1]
    
    for i in [108, 109, 110, 111, 112, 113, 114, 115, 116]:
        rect = plt.Rectangle([anchors_all[i, 0],anchors_all[i, 1]],box_widths[i],box_heights[i],color="r",fill=False)
        ax.add_patch(rect)
    plt.show()
