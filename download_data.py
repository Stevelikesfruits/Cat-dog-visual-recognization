# -*- coding: utf-8 -*-
"""
猫狗数据集下载与整理脚本

功能:
    1. 通过 kagglehub 从 Kaggle 自动下载 Dogs vs Cats 数据集(无需注册账号)
    2. 自动解压(数据集里若包含 zip 包)
    3. 按 "cat" / "dog" 整理图片, 按比例划分为训练集和验证集

最终目录结构(ImageFolder 标准格式, train.py 直接读取):
    data/
    ├── train/
    │   ├── cat/    xxx.jpg ...
    │   └── dog/    xxx.jpg ...
    └── val/
        ├── cat/    xxx.jpg ...
        └── dog/    xxx.jpg ...

用法:
    python download_data.py
"""
import os
import random
import shutil
import zipfile

# Kaggle 上的数据集名称, 下载到本地缓存 (~/.cache/kagglehub)
DATASET = "tongpython/cat-and-dog"

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
TRAIN_RATIO = 0.8   # 80% 训练, 20% 验证
SEED = 42


def label_of(rel_path: str) -> str | None:
    """根据文件所在的相对路径判断属于猫还是狗。

    原版 Kaggle 数据集的图片名是 cat.1.jpg / dog.1.jpg 风格,
    整理过的数据集则是按 cat/ dog/ 目录组织, 这里两种都兼容。
    """
    rel = rel_path.lower().replace("\\", "/")
    base = rel.rsplit("/", 1)[-1]
    parent = rel.rsplit("/", 1)[0].rsplit("/", 1)[-1] if "/" in rel else ""

    # 1. 文件名前缀 (cat.1.jpg 这种)
    if base.startswith("cat."):
        return "cat"
    if base.startswith("dog."):
        return "dog"
    # 2. 所在目录名
    if parent in ("cat", "cats"):
        return "cat"
    if parent in ("dog", "dogs"):
        return "dog"
    # 3. 兜底: 相对路径里只出现一种关键词
    if "cat" in rel and "dog" not in rel:
        return "cat"
    if "dog" in rel and "cat" not in rel:
        return "dog"
    return None


def collect_images(root: str) -> dict:
    """遍历目录, 收集所有猫狗图片, 返回 {"cat": [路径...], "dog": [路径...]}"""
    images = {"cat": [], "dog": []}
    skipped = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if not fn.lower().endswith(IMG_EXTS):
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root)
            label = label_of(rel)
            if label is None:
                skipped += 1
                continue
            images[label].append(full)
    if skipped:
        print(f"[提示] 有 {skipped} 张图片无法判断是猫是狗, 已跳过(不影响使用)")
    return images


def extract_zips(root: str):
    """如果数据集目录里还有 zip 压缩包, 就地解压出来"""
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".zip"):
                zpath = os.path.join(dirpath, fn)
                target = os.path.join(dirpath, fn[:-4])
                if os.path.isdir(target):
                    continue  # 已经解压过
                print(f"解压 {zpath} ...")
                with zipfile.ZipFile(zpath) as zf:
                    zf.extractall(target)


def make_dirs():
    for split in ("train", "val"):
        for label in ("cat", "dog"):
            os.makedirs(os.path.join(DATA_DIR, split, label), exist_ok=True)


def split_and_copy(images: dict):
    """按比例切分训练/验证集, 并复制图片到对应目录"""
    random.seed(SEED)
    total = {"cat": 0, "dog": 0}
    for label, paths in images.items():
        random.shuffle(paths)
        n_train = int(len(paths) * TRAIN_RATIO)
        for i, src in enumerate(paths):
            split = "train" if i < n_train else "val"
            dst = os.path.join(DATA_DIR, split, label, os.path.basename(src))
            # 不同子目录可能有同名文件, 加序号前缀避免覆盖
            if os.path.exists(dst):
                dst = os.path.join(
                    os.path.dirname(dst),
                    f"{i:05d}_{os.path.basename(src)}",
                )
            shutil.copy2(src, dst)
            total[label] += 1
        print(f"[{label}] 共 {len(paths)} 张: 训练 {n_train} 张, 验证 {len(paths) - n_train} 张")
    return total


def main():
    print(f"数据集目录: {DATA_DIR}")
    if os.path.isdir(os.path.join(DATA_DIR, "train", "cat")):
        n_cat = len(os.listdir(os.path.join(DATA_DIR, "train", "cat")))
        if n_cat > 0:
            print(f"[提示] 数据已存在({n_cat}+ 张猫图), 如需重新下载请先删除 data/ 目录")
            return

    print(f"正在从 Kaggle 下载数据集 [{DATASET}] ...")
    try:
        import kagglehub
        root = kagglehub.dataset_download(DATASET)
    except Exception as e:
        print(f"\n[下载失败] {e}")
        print("可能原因: 网络不通或 Kaggle 访问较慢, 请稍后重试。")
        print("备选方案: 手动下载 dogs-vs-cats 数据集后, 把图片按 cat/dog 目录放好")
        print("          (目录结构见本文件顶部注释), 再直接运行 train.py 即可。")
        return

    print(f"下载完成, 原始文件位于: {root}")
    extract_zips(root)

    print("正在整理图片 ...")
    images = collect_images(root)
    if not images["cat"] or not images["dog"]:
        print("[错误] 未找到足够的猫/狗图片, 请检查数据集来源")
        return

    make_dirs()
    split_and_copy(images)

    n_cat_train = len(os.listdir(os.path.join(TRAIN_DIR, "cat")))
    n_dog_train = len(os.listdir(os.path.join(TRAIN_DIR, "dog")))
    n_cat_val = len(os.listdir(os.path.join(VAL_DIR, "cat")))
    n_dog_val = len(os.listdir(os.path.join(VAL_DIR, "dog")))
    print("\n整理完成!")
    print(f"  训练集: 猫 {n_cat_train} 张, 狗 {n_dog_train} 张")
    print(f"  验证集: 猫 {n_cat_val} 张, 狗 {n_dog_val} 张")
    print("接下来运行: python train.py")


if __name__ == "__main__":
    main()
