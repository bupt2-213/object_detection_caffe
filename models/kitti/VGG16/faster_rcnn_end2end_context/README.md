# 模型说明

1. 训练数据集：KITTI
2. 模型：Faster RCNN 端到端模型
3. 说明：

    * Faster RCNN Baseline
    * conv5-3 输入 RPN 网络之前经过了一个多分支上下文模块（1x1, 3x3, 5x5卷积分支）
    
4. 部分主要参数：

    * `feat_stride`: 16,
    * `scales`: [2, 4, 8, 16, 32], 
    * `base_size`: 16, 
    * `ratios`: [0.5, 1, 2]