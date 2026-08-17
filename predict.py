# -*- coding: utf-8 -*-
"""
猫狗识别 —— 预测脚本

用法:
    python predict.py 图片路径          # 判断单张图片
    python predict.py 图片文件夹路径     # 批量判断文件夹里的所有图片

示例:
    python predict.py data/val/cat/00000_xxx.jpg
    python predict.py data/val          # 批量判断整个验证集
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# 与 train.py 保持一致的模型结构定义
from train import CatDogCNN

MODEL_PATH = "catdog_cnn.pt"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
IMG_SIZE = 128
CLASS_NAMES = ["cat", "dog"]  # 与 ImageFolder 的目录顺序一致


def make_transform():
    return transforms.Compose([
        transforms.Resize(IMG_SIZE + 16),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def load_model(device):
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"[错误] 找不到模型文件 {MODEL_PATH}, 请先运行 python train.py")
    model = CatDogCNN(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()
    return model


def predict_image(model, transform, path, device):
    """返回 (类别名, 置信度)"""
    img = Image.open(path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        prob = torch.softmax(logits, dim=1)[0]  # softmax 把 logits 变成概率
    idx = int(prob.argmax())
    return CLASS_NAMES[idx], float(prob[idx])


def show_image(path, label, conf):
    """用 matplotlib 显示图片和预测结果(可选的)"""
    import numpy as np
    img = np.asarray(Image.open(path).convert("RGB"))
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img)
    ax.axis("off")
    cn = "猫" if label == "cat" else "狗"
    ax.set_title(f"预测: {cn}  (置信度 {conf:.1%})")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="猫狗识别预测")
    parser.add_argument("path", help="图片路径或包含图片的文件夹路径")
    parser.add_argument("--show", action="store_true", help="弹窗显示图片")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    transform = make_transform()

    if os.path.isdir(args.path):
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        files = sorted(f for f in os.listdir(args.path)
                       if f.lower().endswith(exts))
        if not files:
            sys.exit(f"[错误] 文件夹 {args.path} 里没有图片")
        n_cat = n_dog = 0
        for fn in files:
            full = os.path.join(args.path, fn)
            label, conf = predict_image(model, transform, full, device)
            n_cat += label == "cat"
            n_dog += label == "dog"
            # 注意: 不要在终端里打印 emoji, Windows 默认 GBK 编码会报错
            print(f"{fn:40s} → {'猫' if label == 'cat' else '狗'} ({conf:.1%})")
        print(f"\n统计: 猫 {n_cat} 张, 狗 {n_dog} 张, 共 {len(files)} 张")
    else:
        if not os.path.exists(args.path):
            sys.exit(f"[错误] 文件不存在: {args.path}")
        label, conf = predict_image(model, transform, args.path, device)
        print(f"{os.path.basename(args.path)} → {'猫' if label == 'cat' else '狗'} (置信度 {conf:.1%})")
        if args.show:
            show_image(args.path, label, conf)


if __name__ == "__main__":
    main()
