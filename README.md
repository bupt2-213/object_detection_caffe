# Object Detection caffe


说明：

1. caffe 环境的编译和配置参考 README.md.origin 文件。
2. 本项目主要基于 VGG16 来做的，各个模型配置参考 models 文件夹，对应的文件夹下 README.md 有具体的说明。
3. 机器环境：

    * Ubuntu 16.04
    * CUDA 8.0
    * cudnn 5.1
    * GPU 1050ti(4G)
    * opencv 4.0

4. 由于受到显存的限制，本项目在训练模型的时候使用了较小的输入图像尺寸。