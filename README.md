## CIFAR10上的简单深度神经网络

​		本仓库使用Pytorch实现了一个简单的深度神经网络和ResNet网络的复现，并且在CIFAR10数据集上进行训练和测试，分别得到了在测试集的准确度。

​		并且在model中提供了简单CNN模型和ResNet18模型的预训练模型。



## 目录

- 介绍
- 安装
- 测试
- 评估
- 结论







## 介绍

### CIFAR10

​		CIFAR-10是一个彩色图像数据集。CIFAR-10 是由Hinton 的学生Alex Krizhevsky 和Ilya Sutskever 整理的一个用于识别普适物体的小型数据集。一共包含10 个类别的RGB 彩色图片：飞机（ airplane ）、汽车（ automobile ）、鸟类（ bird ）、猫（ cat ）、鹿（ deer ）、狗（ dog ）、蛙类（ frog ）、马（ horse ）、船（ ship ）和卡车（ truck ）。其中每个图片的尺寸为32 × 32 ，每个类别有6000个图像，数据集中一共有50000 张训练图片和10000 张测试图片。

​		CIFAR-10 含有的是现实世界中真实的物体，不仅噪声很大，而且物体的比例、特征都不尽相同，这为识别带来很大困难。由于数据集中每张图像为32x32，包含有RGB3个通道，按照RGB通道顺序以及每一通道按照行的顺序已排列好，一个训练样本对应一行有32x32x3=3072个值。

![output](picture\output.png)



### ResNet

​		ResNet是残差神经网络，我们知道，网络越深，咱们能获取的信息越多，而且特征也越丰富。但是根据实验表明，随着网络的加深，优化效果反而越差，测试数据和训练数据的准确率反而降低了。这是因为网络的加深造成的梯度爆炸和梯度消失的问题。ResNet网络是参考了VGG19网络，在其基础上进行了修改，并通过类似短路的机制加入了残差单元，使得神经网络拥有残差学习。



<div>
    <img src="picture\image-20221010100438802.png" width=500 height=500>


## 安装

### 1. 克隆代码

```
git clone https://github.com/20184205/CIFAR10_easy_model.git
```



### 2. 安装依赖

```
conda create -n pytorchcpu python=3.9
source activate pytorchcpu
# 使用conda安装requirements文件
conda install --yes --file requirements.txt
```



### 3. 训练模型

```
# 简单CNN网络
python script/train_model.py -e 30 -l 1e-3 -s model/cnn_model.kpl
# ResNet18
python script/train_model.py -m ResNet -e 30 -l 1e-3 -s model/resnet_model.kpl
```



## 测试

```
# 测试简单CNN网络
python script/test_model.py -m CNN -l model/cnn_model.kpl
# 测试ResNet18
python script/test_model.py -m ResNet -l model/res_model.kpl
```



## 评估

|          | 10 epoch | 40 epoch | 80 epoch |
| -------- | -------- | -------- | -------- |
| CNN      | 73.4%    | 72.3%    | 73.4%    |
| ResNet18 | 77.3%    | 80.1%    | 81.6%    |

