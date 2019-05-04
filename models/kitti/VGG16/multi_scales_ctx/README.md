# 模型说明

1. 训练数据集：KITTI
2. 模型：Faster RCNN + multi-scale + ROI context
3. 说明：

    * Faster RCNN Baseline
    * conv4-2 和 conv4-3 融合 conv4-fuse，con5-2 和 conv5-3 融合得到 conv5-fuse，然后将 conv5-fuse 上采样并与 conv4-fuse 融合，融合特征作为最终卷积层的输出。
    * ROI 输出经过了一个 ROI 上下文模块，为两倍区域部分。
