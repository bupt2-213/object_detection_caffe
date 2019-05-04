# 模型说明

1. 训练数据集：KITTI
2. 模型：Faster RCNN + multi-scale
3. 说明：

    * Faster RCNN Baseline
    * conv4-2 和 conv4-3 融合 conv4-fuse，con5-2 和 conv5-3 融合得到 conv5-fuse，然后将 conv5-fuse 上采样并与 conv4-fuse 融合，融合特征作为最终卷积层的输出。
