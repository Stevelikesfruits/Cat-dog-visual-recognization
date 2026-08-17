# -*- coding: utf-8 -*-
"""
猫狗识别 —— 自建 CNN 训练脚本

模型结构: Conv → BN → ReLU 的卷积块 ×3 + 全局平均池化 + 全连接分类头
数据增强: 随机裁剪缩放 / 随机水平翻转 / 颜色抖动
优化器:   Adam (带 L2 正则化, 即 weight_decay)
评价:     每轮在验证集上评估, 自动保存验证准确率最高的模型

用法:
    python train.py                          # 完整训练
    python train.py --epochs 5               # 只训练 5 轮
    python train.py --limit 200 --epochs 1   # 小规模冒烟测试(快速验证代码能跑通)
"""
import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")  # 只保存图片不弹窗, 避免训练时被窗口阻塞
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ======================= 超参数(可自行调整) =======================
IMG_SIZE = 128          # 输入图片统一缩放到 128×128
BATCH_SIZE = 64         # 每批图片数量(显存不够就调小)
EPOCHS = 12             # 训练轮数
LR = 1e-3               # 学习率
WEIGHT_DECAY = 1e-4     # L2 正则化强度(weight_decay), 你笔记里学过的
SEED = 42               # 随机种子, 保证结果可复现
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SAVE_PATH = "catdog_cnn.pt"          # 模型保存位置(.pt 已被 .gitignore 忽略)
OUTPUT_DIR = "outputs"               # 训练曲线图保存位置

# 归一化用的经验均值/方差(ImageNet 统计值, 对自建网络也适用)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


# ======================= 模型定义 =======================
class ConvBlock(nn.Module):
    """一个卷积块: Conv → BN → ReLU → Conv → BN → ReLU → MaxPool

    和理论笔记对应:
      - 卷积层: 提取局部特征
      - BN: 标准化, 加速收敛
      - ReLU: 非线性激活
      - MaxPool: 下采样, 扩大感受野
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 尺寸减半
        )

    def forward(self, x):
        return self.block(x)


class CatDogCNN(nn.Module):
    """猫狗二分类网络: 3 个卷积块 + 全局平均池化 + 2 层全连接

    输入: [N, 3, 128, 128]  输出: [N, 2]  (猫/狗的 logits)
    说明: PyTorch 默认用 Kaiming(MSRA)初始化卷积层权重,
          就是你笔记里学过的 MSRA 初始化方法。
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),    # 128→64
            ConvBlock(32, 64),   # 64→32
            ConvBlock(64, 128),  # 32→16
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化 → [N, 128, 1, 1]
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),     # dropout 随机丢弃神经元, 抑制过拟合
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),  # 输出 2 个类的 logits
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ======================= 数据处理 =======================
def make_transforms(train: bool):
    """训练/验证用不同的预处理。

    训练集加数据增强(随机裁剪、翻转、颜色抖动), 相当于扩充了数据集,
    能有效抑制过拟合 —— 也是你笔记里学过的手法。
    """
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.Resize(IMG_SIZE + 16),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def load_data(batch_size: int, limit: int = 0):
    """加载 data/train 和 data/val, 返回 train_loader, val_loader, class_names"""
    if not os.path.isdir(os.path.join(DATA_DIR, "train")):
        raise FileNotFoundError(
            "未找到 data/ 目录, 请先运行: python download_data.py"
        )
    train_ds = datasets.ImageFolder(
        os.path.join(DATA_DIR, "train"), transform=make_transforms(train=True)
    )
    val_ds = datasets.ImageFolder(
        os.path.join(DATA_DIR, "val"), transform=make_transforms(train=False)
    )
    classes = train_ds.classes  # 先记录类别名, 后面可能被 Subset 包一层
    if limit > 0:  # 冒烟测试: 只取前 limit 张
        train_ds = torch.utils.data.Subset(train_ds, range(min(limit, len(train_ds))))
        val_ds = torch.utils.data.Subset(val_ds, range(min(limit, len(val_ds))))
    print(f"训练集 {len(train_ds)} 张, 验证集 {len(val_ds)} 张, 类别: {classes}")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    return train_loader, val_loader, classes


# ======================= 训练流程 =======================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total, correct, running_loss = 0, 0, 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()   # 反向传播, 你笔记里学过的自动求导
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        total += labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total, correct, running_loss = 0, 0, 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        total += labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
    return running_loss / total, correct / total


def plot_curves(history: dict):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history["train_loss"], "o-", label="train")
    axes[0].plot(epochs, history["val_loss"], "s-", label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(epochs, history["train_acc"], "o-", label="train")
    axes[1].plot(epochs, history["val_acc"], "s-", label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "training_curves.png")
    fig.savefig(path, dpi=120)
    print(f"训练曲线已保存: {path}")


def main():
    parser = argparse.ArgumentParser(description="猫狗识别训练")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=0, help="只取前 N 张做冒烟测试")
    parser.add_argument("--lr", type=float, default=LR)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader, val_loader, class_names = load_data(args.batch_size, args.limit)

    model = CatDogCNN(num_classes=len(class_names)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params / 1e3:.1f}K")
    print(model)

    criterion = nn.CrossEntropyLoss()          # 分类任务的标准损失
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    )  # 验证损失不再下降时, 学习率减半

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc, best_epoch = 0.0, -1
    print(f"\n开始训练, 共 {args.epochs} 轮 ...\n")
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(va_loss)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        print(f"[轮次 {epoch:2d}] 训练 loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"验证 loss={va_loss:.4f} acc={va_acc:.4f} | 耗时 {time.time() - t_epoch:.0f}s")
        if va_acc > best_acc:
            best_acc, best_epoch = va_acc, epoch
            torch.save(model.state_dict(), SAVE_PATH)  # 只保存表现最好的模型
            print(f"          → 验证准确率提升, 模型已保存到 {SAVE_PATH}")

    print(f"\n训练完成, 总耗时 {time.time() - t_start:.0f}s")
    print(f"最佳验证准确率: {best_acc:.4f} (第 {best_epoch} 轮)")
    plot_curves(history)
    print(f"接下来可以用: python predict.py 某张图片的路径")


if __name__ == "__main__":
    main()
