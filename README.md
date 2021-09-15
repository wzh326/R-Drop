# R-Drop
# 一.简介
一般模型在引入正则化项dropout后，每次训练都会随机丢弃部分神经元，这相当于每次训练的都是原模型的一个子模型。经过无数次训练之后，在预测阶段启用全部神经元，将所有经过训练的子模型耦合成一个大的分类器，这可以一定程度上提高模型的泛化能力。
但是一副图像的不同细节随机组合得到的不同特征所各自表示的实际意义很有可能天差地别，最后得到的概率分布自然也不一样。强行将这些概率分布融合到一起并不见得就是件好事。这在某种程度上增加了模型的不稳定性。而KL散度是一种判断两个概率分布之间差距的评价标准。通过在损失函数中增加KL散度，可以约束两个子模型所输出的概率分布。这样可以使得在利用模型不同特征的时候还尽量保证所有子模型对图像类别的判断不至于背道而驰，在充分利用dropout所具有的泛化优势的时候也一定程度上克服了缺点，增加了模型的稳定性。
![1](https://user-images.githubusercontent.com/79301727/133238064-332c21c8-d2a9-465d-aa15-2c60d887e12d.jpg)
# 二.复现精度
|准确率|原论文|复现|
|--|--|--|
|R-Drop|93.29|93.29|
# 三.数据集
本次复现采用的数据集是经典数据集cifar-100，在网上可以很轻松的下载到http://www.cs.toronto.edu/~kriz/cifar.html。
在paddle内置的datasets中也有此数据集。
数据集大小：60000个32x32彩色图像组成，50000个训练集，10000个预测集
数据集格式：每一张图片吧都有五个信息filenames、batch_label、fine_labels、coarse_labels和data，分别是图像名称、batch_label、细粒度标签、粗粒度标签、图像表示的信息。
# 四.环境依赖
硬件：所有支持paddlepaddle的硬件  
框架：paddlepaddle>=2.1.0
其他依赖库：  
ml_collections
# 五.快速开始
1.首先安装依赖  
2.克隆项目 
```
!git clone https://github.com/wzh326/R-Drop/  
```
3.下载预训练模型
```
%cd 
!wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz
```
4.开始训练。
```
!python3 train.py --name cifar100-100_500 --dataset cifar100 --model_type ViT-B_16 --pretrained_dir ViT-B_16.npz
```
5.用自己训练的模型预测
```
!python3 eval.py --model {自己训练的模型路径} 
```
6.用预训练模型预测，最后两个参数是存储预训练模型的路径的一部分
```
!python3 eval.py --model_load True --output_dir output_dir/ --name cifar100-100_500
```
# 代码结构说明
model文件夹下是模型结构，modeiling ViT主体部分，modeling_resnat
output_dir下面是我们训练好的模型
# 模型信息
|项目作者|白告|
|--|--|
|项目大小|600M+|
|--|--|
|模型结构|ViB-T/16+RD|
|--|--|
|飞桨地址|https://aistudio.baidu.com/aistudio/projectdetail/2353720|
|--|--|
|模型目的|图像分类|
|--|--|
|数据集|cifar100|


