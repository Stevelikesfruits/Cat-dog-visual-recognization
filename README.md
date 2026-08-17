# 猫狗图像识别 (PyTorch)

用 PyTorch 从零搭建卷积神经网络 (CNN) 进行猫狗二分类。

## 环境

- conda 环境: `pytorch_env` (`D:\Anaconda\envs\pytorch_env`)
- 核心库: torch 2.13.0+cu126, torchvision, numpy, pillow (已装好)
- 额外库: `kagglehub` (自动下载数据集), `matplotlib` (画训练曲线)

## 使用步骤

激活环境后(或在 PyCharm 中选择该解释器), 依次运行:

```bash
# 1. 下载数据集 (从 Kaggle 自动下载, 无需账号, 约几百 MB)
python download_data.py

# 2. 训练 (RTX 4060 上约 10~20 分钟, 验证准确率约 85%+)
python train.py

# 3. 预测: 判断一张图片 / 批量判断一个文件夹
python predict.py 某张图片.jpg
python predict.py data/val/dog
```

## 文件说明

| 文件 | 作用 |
|---|---|
| `download_data.py` | 下载 Dogs vs Cats 数据集, 整理成 `data/train` 和 `data/val` 的 cat/dog 目录 |
| `train.py` | 定义 CNN 模型, 训练并保存最优模型 `catdog_cnn.pt`, 画出训练曲线 |
| `predict.py` | 加载模型, 对单张图片或整个文件夹做预测 |

## 模型结构

```
输入 3×128×128 图片
Conv(3→32) → BN → ReLU → Conv → BN → ReLU → MaxPool
Conv(32→64) → BN → ReLU → Conv → BN → ReLU → MaxPool
Conv(64→128) → BN → ReLU → Conv → BN → ReLU → MaxPool
AdaptiveAvgPool → Dropout(0.5) → 全连接(128→64) → ReLU → 全连接(64→2)
```

对应理论笔记: 卷积特征提取 / BN 标准化 / ReLU 激活 / MaxPool 下采样 /
Dropout 与 L2 正则化(weight_decay) / Adam 优化器 / CrossEntropy 分类损失。

## 常用调整

- 想训练更久提高准确率: `python train.py --epochs 20`
- 显存不够: `python train.py --batch-size 32`
- 快速验证代码能跑通: `python train.py --limit 200 --epochs 1`

## 常见问题

- **下载失败/太慢**: 网络问题, 稍后重试。也可手动下载 dogs-vs-cats 数据集,
  自己按 `data/train/{cat,dog}`、`data/val/{cat,dog}` 放好图片后直接训练。
- **`data/` 目录不会被提交进 git**(已在 .gitignore 中), 模型文件同理。
