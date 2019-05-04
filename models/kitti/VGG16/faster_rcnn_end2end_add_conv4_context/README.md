# 模型说明

1. 训练数据集：KITTI
2. 模型：Faster RCNN 端到端模型
3. 说明：

    * Faster RCNN Baseline
    * 将 conv4-3 下采样，得到与 conv5-3 分辨率相同的特征图，然后进行融合，将融合后的特征图作为 RPN 网络的输入。
    * conv4-3 下采样之前经过了一个多分支上下文模块（1x1, 3x3, 5x5 的卷积分支）
    
4. 部分主要参数：

    * `feat_stride`: 16,
    * `scales`: [2, 4, 8, 16, 32], 
    * `base_size`: 16, 
    * `ratios`: [0.5, 1, 2]