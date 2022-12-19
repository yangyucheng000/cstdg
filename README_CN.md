# 目录

- [目录](#contents)
    - [CTSDG描述](#ctsdg-description)
    - [Model框架结构](#model-architecture)
    - [数据集](#dataset)
    - [环境需求](#environment-requirements)
    - [快速开始](#quick-start)
    - [脚本描述](#script-description)
        - [脚本和示例代码](#script-and-sample-code)
        - [脚本参数](#script-parameters)
        - [训练流程](#training-process)
        - [测试流程](#evaluation-process)
        - [输出MINDIR](#export-mindir)
    - [模型描述](#model-description)
        - [在GPU上的训练表现](#training-performance-gpu)
    - [随机情况描述](#description-of-random-situation)
    - [ModelZoo主页](#modelzoo-homepage)

## [CTSDG description](#contents)

近年来，通过引入结构先验知识，深度生成方法在图像修复领域取得了长足进展。然而，由于在结构重建过程中缺乏与图像纹理的适当交互，现有的解决方案无法处理具有较大腐蚀的情况，并且通常会导致结果失真。这是一种新颖的用于图像修复的双流网络，它以耦合方式对结构约束的纹理合成和纹理引导的结构重建进行建模，以便它们更好地相互利用，生成更合理的图像。此外，为了增强全局一致性，设计了双向选通特征融合（Bi-GFF）模块来交换和组合结构和纹理信息，并开发了上下文特征聚合（CFA）模块来通过区域关联学习和多尺度特征聚合来细化生成的内容。

> [沦为](https://arxiv.org/pdf/2108.09760.pdf):  Image Inpainting via Conditional Texture and Structure Dual Generation
> Xiefan Guo, Hongyu Yang, Di Huang, 2021.
> [补充资料](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Guo_Image_Inpainting_via_ICCV_2021_supplemental.pdf)

## [模型框架结构](#contents)

## [数据集](#contents)

使用的数据集: [CELEBA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), [NVIDIA Irregular Mask Dataset](https://nv-adlr.github.io/publication/partialconv-inpainting)

- 你需要从 **CELEBA** 下载 (板块 *Downloads -> Align&Cropped Images*):
    - `img_align_celeba.zip`
    - `list_eval_partitions.txt`
- 你需要从 **NVIDIA Irregular Mask Dataset** 下载:
    - `irregular_mask.zip`
    - `test_mask.zip`
- 目录结构如下:

  ```text
    .
    ├── img_align_celeba            # 图像文件夹
    ├── irregular_mask              # 用于训练的掩盖
    │   └── disocclusion_img_mask
    ├── mask                        # 用于测试的掩盖
    │   └── testing_mask_dataset
    └── list_eval_partition.txt     # train/val/test 拆分
  ```

## [环境需求](#contents)

- 硬件（GPU）
    - 使用GPU处理器准备硬件环境。
- 框架
    - [MindSpore1.8.1](https://gitee.com/mindspore/mindspore)
- 有关更多信息，请查看以下资源：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)
- 下载数据集

## [快速开始](#contents)

### [预训练VGG16](#contents)

您需要将火炬VGG16模型转换为训练CTSDG模型的感知损失。

1. [下载预训练VGG16](https://download.pytorch.org/models/vgg16-397923af.pth)
2. 将torch模型转换为mindspore:

```shell
python converter.py --torch_pretrained_vgg=/path/to/torch_pretrained_vgg
```

转换后的mindspore检查点将与名为“vgg16_feat_extr_ms.ckpt”的torch模型保存在同一目录中。

准备好数据集并转换VGG16后，您可以开始训练和评估，如下所示：

### [在GPU上运行](#contents)

#### 训练

```shell
# 单设备训练
bash scripts/run_standalone_train_gpu.sh [DEVICE_ID] [CFG_PATH] [SAVE_PATH] [VGG_PRETRAIN] [IMAGES_PATH] [MASKS_PATH] [ANNO_PATH]

# 分布式训练
bash scripts/run_distribute_train_gpu.sh [DEVICE_NUM] [CFG_PATH] [SAVE_PATH] [VGG_PRETRAIN] [IMAGES_PATH] [MASKS_PATH] [ANNO_PATH]
```

示例:

```shell
# 单设备训练
# DEVICE_ID - 设备ID
# CFG_PATH - 配置路径
# SAVE_PATH - 保存日志和模型的路径
# VGG_PRETRAIN - 已预训练VGG16的路径
# IMAGES_PATH - CELEBA路径
# MASKS_PATH - 用于训练掩盖的路径
# ANNO_PATH - 用于train/val/test文件·路径
bash scripts/run_standalone_train_gpu.sh 0 ./default_config.yaml /path/to/output /path/to/vgg16_feat_extr.ckpt /path/to/img_align_celeba /path/to/training_mask /path/to/list_eval_partitions.txt

# 分布式训练 (8p)
# DEVICE_NUM - 训练设备的数量
# 其他参数跟独立训练的参数一样
bash scripts/run_distribute_train_gpu.sh 8 ./default_config.yaml /path/to/output /path/to/vgg16_feat_extr.ckpt /path/to/img_align_celeba /path/to/training_mask /path/to/list_eval_partitions.txt
```

#### 评测

```shell
# 评测
bash scripts/run_eval_gpu.sh [DEVICE_ID] [CFG_PATH] [CKPT_PATH] [IMAGES_PATH] [MASKS_PATH] [ANNO_PATH]
```

Example:

```shell
# evaluate
# DEVICE_ID - 设备ID
# CFG_PATH - 配置路径
# CKPT_PATH - 用于评估的ckpt文件路径
# IMAGES_PATH - img_align_celeba数据集路径
# MASKS_PATH - 用于测试掩盖的路径
# ANNO_PATH - 用于train/val/test拆分文件的路径
bash scripts/run_eval_gpu.sh 0 ./default_config.yaml /path/to/ckpt /path/to/img_align_celeba /path/to/testing_mask /path/to/list_eval_partitions.txt  
```

## [脚本描述](#contents)

### [脚本和示例代码](#contents)

```text
 .
 ├── configs
 │   └── jester_config.yaml               #用于在GPU上训练的参数配置
 ├── model_utils                          #ModelArts工具
 │   ├── __init__.py
 │   ├── config.py                        #读取参数配置
 │   ├── device_adapter.py                #设备适应
 │   ├── local_adapter.py                 #本地适应
 │   ├── logging.py	                     #获得日志
 │   ├── moxing_adapter.py		       #适配ModelArts
 │   └── util.py
 ├── scripts
 │   ├── convert_bn_inception.sh
 │   ├── preprocess_jester_dataset.sh
 │   ├── run_distributed_train_gpu.sh
 │   ├── run_eval_gpu.sh
 │   ├── run_export_gpu.sh
 │   ├── run_standalone_train_gpu.sh
 │   └── unpack_jester_dataset.sh
 ├── src
 │   ├── __init__.py
 │   ├── bn_inception.py                 #读取bn_inception预训练模型
 │   ├── convert_bn_inception.py         #转化bn_inception模型
 │   ├── preprocess_jester_dataset.py    #处理
 │   ├── train_cell.py			 #训练神经元
 │   ├── transforms.py                   #模型转化
 │   ├── trn.py   		         #trn神经网络
 │   ├── tsn.py                          #tsn神经网络
 │   ├── tsn_dataset.py                  #处理tsn数据集
 │   └── utils.py
 ├── eval.py				 #测试模型
 ├── export.py                           #输出模型
 ├── README.md             	         #TRN模型说明
 ├── requirements.txt			 #需求文件
 └── train.py			        	#训练模型
```

### [脚本参数](#contents)

训练参数可以在`default_config.yaml`中配置

```text
"gen_lr_train": 0.0002,                     # 用于生成器训练的学习率
"gen_lr_finetune": 0.00005,                 # 用于生成器微调的学习率
"dis_lr_multiplier": 0.1,                   # 鉴别器的学习率是生成器的学习率乘以该参数
"batch_size": 6,                            # 批次大小
"train_iter": 350000,                       # 训练迭代器数量
"finetune_iter": 150000                     # 微调迭代器数量
"image_load_size": [256, 256]               # 输入图像大小
```

有关更多参数，请参阅“default_config.yaml”的内容

### [训练流程](#contents)

#### [在GPU上运行](#contents)

##### 单设备运行 (1p)

```shell
# DEVICE_ID - 设备ID (0)
# CFG_PATH - 配置路径 (./default_config.yaml)
# SAVE_PATH - 保存日志和模型的路径 (/path/to/output)
# VGG_PRETRAIN - 已预训练VGG16的路径 (/path/to/vgg16_feat_extr.ckpt)
# IMAGES_PATH - CELEBA数据集路径 (/path/to/img_align_celeba)
# MASKS_PATH - 用于训练掩盖的路径 (/path/to/training_mask)
# ANNO_PATH - 用于train/val/test拆分文件的路径 (/path/to/list_eval_partitions.txt)
bash scripts/run_standalone_train_gpu.sh 0 ./default_config.yaml /path/to/output /path/to/vgg16_feat_extr.ckpt /path/to/img_align_celeba /path/to/training_mask /path/to/list_eval_partitions.txt
```

日志将会储存在路径`/path/to/output/log.txt`

结果:

```text
...
DATE TIME iter: 250, loss_g: 19.7810001373291, loss_d: 1.7710000276565552, step time: 570.67 ms
DATE TIME iter: 375, loss_g: 20.549999237060547, loss_d: 1.8650000095367432, step time: 572.09 ms
DATE TIME iter: 500, loss_g: 25.295000076293945, loss_d: 1.8630000352859497, step time: 572.23 ms
DATE TIME iter: 625, loss_g: 24.059999465942383, loss_d: 1.812999963760376, step time: 573.33 ms
DATE TIME iter: 750, loss_g: 26.343000411987305, loss_d: 1.8539999723434448, step time: 573.18 ms
DATE TIME iter: 875, loss_g: 21.774999618530273, loss_d: 1.8509999513626099, step time: 573.0 ms
DATE TIME iter: 1000, loss_g: 18.062999725341797, loss_d: 1.7960000038146973, step time: 572.41 ms
...
```

##### 分布式训练 (8p)

```shell
# DEVICE_NUM - 设备ID (8)
# 其他参数跟独立训练的参数一样
bash scripts/run_distribute_train_gpu.sh 8 ./default_config.yaml /path/to/output /path/to/vgg16_feat_extr.ckpt /path/to/img_align_celeba /path/to/training_mask /path/to/list_eval_partitions.txt
```

日志将会储存在路径`/path/to/output/log.txt`

结果:

```text
...
DATE TIME iter: 250, loss_g: 26.28499984741211, loss_d: 1.680999994277954, step time: 757.67 ms
DATE TIME iter: 375, loss_g: 21.548999786376953, loss_d: 1.468000054359436, step time: 758.02 ms
DATE TIME iter: 500, loss_g: 17.89299964904785, loss_d: 1.2829999923706055, step time: 758.57 ms
DATE TIME iter: 625, loss_g: 18.750999450683594, loss_d: 1.2589999437332153, step time: 759.95 ms
DATE TIME iter: 750, loss_g: 21.542999267578125, loss_d: 1.1829999685287476, step time: 759.45 ms
DATE TIME iter: 875, loss_g: 27.972000122070312, loss_d: 1.1629999876022339, step time: 759.62 ms
DATE TIME iter: 1000, loss_g: 18.03499984741211, loss_d: 1.159000039100647, step time: 759.51 ms
...
```

### [测评结果](#contents)

#### GPU

```shell
bash scripts/run_eval_gpu.sh [DEVICE_ID] [CFG_PATH] [CKPT_PATH] [IMAGES_PATH] [MASKS_PATH] [ANNO_PATH]
```

示例:

```shell
# DEVICE_ID - 设备ID (0)
# CFG_PATH - 配置路径 (./default_config.yaml)
# CKPT_PATH - 用于评估模型的路径 (/path/to/ckpt)
# IMAGES_PATH - img_align_celeba数据集路径 (/path/to/img_align_celeba)
# MASKS_PATH - 测试掩盖数据集路径 (/path/to/testing/mask)
# ANNO_PATH - 用于train/val/test分割文件路径 (/path/to/list_eval_partitions.txt)
bash scripts/run_eval_gpu.sh 0 ./default_config.yaml /path/to/ckpt /path/to/img_align_celeba /path/to/testing_mask /path/to/list_eval_partitions.txt
```

日志将会储存在路径 `./logs/eval_log.txt`.

结果:

```text
PSNR:
0-20%: 
20-40%: 
40-60%: 
SSIM:
0-20%: 
20-40%: 
40-60%: 
```

### [输出MINDIR](#contents)

如果要推断Ascend 310上的网络，应将模型转换为MINDIR。

#### GPU

```shell
bash scripts/run_export_gpu.sh [DEVICE_ID] [CFG_PATH] [CKPT_PATH]
```

示例:

```shell
# DEVICE_ID - 设备ID (0)
# CFG_PATH - 配置路径 (./default_config.yaml)
# CKPT_PATH - ckpt评估文件路径 (/path/to/ckpt)
bash scripts/run_export_gpu.sh 0 ./default_config.yaml /path/to/ckpt
```

如果要推断Ascend 310上的网络，应将模型转换为MINDIR。`./logs/export_log.txt`, converted model will have the same name as ckpt except extension.

## [模型描述](#contents)

### [在GPU上训练表现](#contents)

| 参数           | CTSDG (1p)                                                                                                                                                                                           |   
|---------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 硬件资源           | GPU: 1*A100, CPU: 8                                                                                                                                                                                           |                                                                                                                                                                                       |
| 更新日期       | 10.15.2022                                                                                                                                                                                                   |                                                                                                                                                                                          |
| Mindspore版本   | 1.8.1                                                                                                                                                                                                        |                                                                                                                                                                                                    |
| 数据集             | CELEBA, NVIDIA Irregular Mask Dataset                                                                                                                                                                        |                                                                                                                                                                    |
| 训练参数 | train_iter=350000, finetune_iter=150000, gen_lr_train=0.0002, gen_lr_finetune=0.00005, dis_lr_multiplier=0.1, batch_size=6                                                                                   |                                                                              
| 优化器           | Adam                                                                                                                                                                                                         |                                                                                                                                                                                                  
| 损失函数       | Reconstruction Loss (L1), Perceptual Loss (L1), Style Loss(L1), Adversarial Loss (BCE), Intermediate Loss (L1 + BCE)                                                                                         |                                                                                       |
| 速度               | 573 ms / step                                                                                                                                                                                                |                                                                                                                                                                                           |
| 精度             | <table><tr><td></td><td>0-20%</td><td>20-40%</td><td>40-60%</td></tr><tr><td>PSNR</td><td>37.79</td><td>29.29</td><td>24.24</td></tr><tr><td>SSIM</td><td>0.979</td><td>0.924</td><td>0.841</td></tr></table> | <table><tr><td></td><td>0-20%</td><td>20-40%</td><td>40-60%</td></tr><tr><td>PSNR</td><td>37.79/td><td>29.29</td><td>24.24</td></tr><tr><td>SSIM</td><td>0.979</td><td>0.924</td><td>0.841</td></tr></table> |

## [随机情况描述](#contents)

`train.py`脚本使用mindspore.set_seed（）设置全局随机种子，可以修改。

## [ModelZoo主页](#contents)

请浏览仓库[homepage](https://gitee.com/he-ruiming/ctsdg).

