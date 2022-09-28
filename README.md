﻿## yolo-tiny目标检测模型 --Pytorch
---

## 目录  
1. [所需环境 Environment](#所需环境) 
2. [注意事项 Attention](#注意事项) 
3. [网络结构 Network Structure](#网络结构)
4. [效果展示 Effect](#效果展示)
5. [数据下载 Download](#数据下载) 
6. [训练步骤 Train](#训练步骤) 

## 所需环境  
1. Python3.7
2. Pytorch>=1.10.1+cu113  
3. Torchvision>=0.11.2+cu113
4. Numpy==1.19.5
5. Pillow==8.2.0
6. Opencv-contrib-python==4.5.1.48
7. onnx==1.12.0
8. onnxruntime-gpu==1.8.0
9. CUDA 11.0+
10. Cudnn 8.0.4+

## 注意事项  
1. 更新yolo-tiny残差模块，加入组卷积(GConv)与通道洗牌机制(ShuffleUnit)  
2．更新基于携带多分类目标掩码的CrossEntropy误差，降低过拟合现象  
3. 更新推理时检测框体的检出标准  
4. 加入正则化操作，降低过拟合影响  
5. 数据与标签路径、训练参数等均位于config.py  
6. onnx通用部署模型转换位于./onnx目录下  

## 网络结构
YOLO based on ShuffleNet  
----------------------------------------------------------------  
        Layer (type)               Output Shape         Param #  
================================================================  
            Conv2d-1         [-1, 32, 208, 208]             864  
       BatchNorm2d-2         [-1, 32, 208, 208]              64  
         ConvBlock-3         [-1, 32, 208, 208]               0  
            Conv2d-4         [-1, 64, 104, 104]          18,432  
       BatchNorm2d-5         [-1, 64, 104, 104]             128  
         ConvBlock-6         [-1, 64, 104, 104]               0  
            Conv2d-7         [-1, 64, 104, 104]          36,864  
       BatchNorm2d-8         [-1, 64, 104, 104]             128  
         ConvBlock-9         [-1, 64, 104, 104]               0  
           Conv2d-10         [-1, 32, 104, 104]           9,216  
      BatchNorm2d-11         [-1, 32, 104, 104]              64  
        ConvBlock-12         [-1, 32, 104, 104]               0  
      ShuffleUnit-13         [-1, 32, 104, 104]               0  
           Conv2d-14         [-1, 32, 104, 104]           9,216  
      BatchNorm2d-15         [-1, 32, 104, 104]              64  
        ConvBlock-16         [-1, 32, 104, 104]               0  
           Conv2d-17         [-1, 64, 104, 104]          36,864  
      BatchNorm2d-18         [-1, 64, 104, 104]             128  
        ConvBlock-19         [-1, 64, 104, 104]               0  
        MaxPool2d-20          [-1, 128, 52, 52]               0  
         ResBlock-21  [[-1, 128, 52, 52], [-1, 64, 104, 104]]               0  
           Conv2d-22          [-1, 128, 52, 52]         147,456  
      BatchNorm2d-23          [-1, 128, 52, 52]             256  
        ConvBlock-24          [-1, 128, 52, 52]               0  
           Conv2d-25           [-1, 64, 52, 52]          36,864  
      BatchNorm2d-26           [-1, 64, 52, 52]             128  
        ConvBlock-27           [-1, 64, 52, 52]               0  
      ShuffleUnit-28           [-1, 64, 52, 52]               0  
           Conv2d-29           [-1, 64, 52, 52]          36,864  
      BatchNorm2d-30           [-1, 64, 52, 52]             128  
        ConvBlock-31           [-1, 64, 52, 52]               0  
           Conv2d-32          [-1, 128, 52, 52]         147,456  
      BatchNorm2d-33          [-1, 128, 52, 52]             256  
        ConvBlock-34          [-1, 128, 52, 52]               0  
        MaxPool2d-35          [-1, 256, 26, 26]               0  
         ResBlock-36  [[-1, 256, 26, 26], [-1, 128, 52, 52]]               0  
           Conv2d-37          [-1, 256, 26, 26]         589,824  
      BatchNorm2d-38          [-1, 256, 26, 26]             512  
        ConvBlock-39          [-1, 256, 26, 26]               0  
           Conv2d-40          [-1, 128, 26, 26]         147,456  
      BatchNorm2d-41          [-1, 128, 26, 26]             256  
        ConvBlock-42          [-1, 128, 26, 26]               0  
      ShuffleUnit-43          [-1, 128, 26, 26]               0  
           Conv2d-44          [-1, 128, 26, 26]         147,456  
      BatchNorm2d-45          [-1, 128, 26, 26]             256  
        ConvBlock-46          [-1, 128, 26, 26]               0  
           Conv2d-47          [-1, 256, 26, 26]         589,824  
      BatchNorm2d-48          [-1, 256, 26, 26]             512  
        ConvBlock-49          [-1, 256, 26, 26]               0  
        MaxPool2d-50          [-1, 512, 13, 13]               0  
         ResBlock-51  [[-1, 512, 13, 13], [-1, 256, 26, 26]]               0  
           Conv2d-52          [-1, 512, 13, 13]       2,359,296  
      BatchNorm2d-53          [-1, 512, 13, 13]           1,024  
        ConvBlock-54          [-1, 512, 13, 13]               0  
         BackBone-55  [[-1, 512, 13, 13], [-1, 256, 26, 26]]               0  
           Conv2d-56            [-1, 32, 26, 1]          16,384  
      BatchNorm2d-57            [-1, 32, 26, 1]              64  
           Conv2d-58           [-1, 512, 13, 1]          16,384  
           Conv2d-59           [-1, 512, 13, 1]          16,384  
        Attention-60          [-1, 512, 13, 13]               0  
           Conv2d-61            [-1, 16, 52, 1]           4,096  
      BatchNorm2d-62            [-1, 16, 52, 1]              32  
           Conv2d-63           [-1, 256, 26, 1]           4,096  
           Conv2d-64           [-1, 256, 26, 1]           4,096  
        Attention-65          [-1, 256, 26, 26]               0  
           Conv2d-66          [-1, 256, 13, 13]         131,072  
      BatchNorm2d-67          [-1, 256, 13, 13]             512  
        ConvBlock-68          [-1, 256, 13, 13]               0  
           Conv2d-69          [-1, 512, 13, 13]       1,179,648  
      BatchNorm2d-70          [-1, 512, 13, 13]           1,024  
        ConvBlock-71          [-1, 512, 13, 13]               0  
           Conv2d-72           [-1, 36, 13, 13]          18,468  
         YoloHead-73           [-1, 36, 13, 13]               0  
           Conv2d-74          [-1, 128, 13, 13]          32,768  
      BatchNorm2d-75          [-1, 128, 13, 13]             256  
        ConvBlock-76          [-1, 128, 13, 13]               0  
         Upsample-77          [-1, 128, 26, 26]               0  
         Upsample-78          [-1, 128, 26, 26]               0  
           Conv2d-79             [-1, 8, 52, 1]           1,024  
      BatchNorm2d-80             [-1, 8, 52, 1]              16  
           Conv2d-81           [-1, 128, 26, 1]           1,024  
           Conv2d-82           [-1, 128, 26, 1]           1,024  
        Attention-83          [-1, 128, 26, 26]               0  
           Conv2d-84          [-1, 256, 26, 26]         884,736  
      BatchNorm2d-85          [-1, 256, 26, 26]             512  
        ConvBlock-86          [-1, 256, 26, 26]               0  
           Conv2d-87           [-1, 36, 26, 26]           9,252  
         YoloHead-88           [-1, 36, 26, 26]               0  
================================================================  
Total params: 6,640,728  
Trainable params: 6,640,728  
Non-trainable params: 0  
----------------------------------------------------------------  
Input size (MB): 1.98  
Forward/backward pass size (MB): 9922.07  
Params size (MB): 25.33  
Estimated Total Size (MB): 9949.38  
----------------------------------------------------------------  
## 效果展示  
![image](https://github.com/JJASMINE22/yolo_tiny_pytorch/blob/main/sample/sample.jpg)  

## 数据下载    
coco2017  
链接：https://cocodataset.org/#home  
下载解压后将数据集放置于config.py中指定的路径。 

## 训练步骤  
运行train.py  
 
