# 模型说明

1. 训练数据集：KITTI
2. 模型：Faster RCNN + ida
3. 说明：

    * Faster RCNN Baseline
    * 将 conv3-3 下采样，得到与 conv4-3 分辨率相同的特征图，然后进行融合。
    * 将融合后的特征图继续下采样，得到与 conv5-3 分辨率相同的特征图，然后再融合，得到最终卷积层输出特征图。
