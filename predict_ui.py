# -*- coding: utf-8 -*-
"""
猫狗识别 —— 图形界面结果展示 (Tkinter)

界面布局:
    ┌──────────────────────────────────────────────┐
    │ [选择文件夹] [选择图片]   状态信息            │
    ├───────────────────┬──────────────────────────┤
    │  图片预览 (左列)    │  预测结果 (右列)         │
    │  缩略图网格,滚轮滚动 │  文件名 → 猫/狗+置信度   │
    │                    │  绿=有把握 橙=一般 红=猜 │
    └───────────────────┴──────────────────────────┘

依赖:
    tkinter (Python 自带, 无需安装)
    pillow  (已安装, 用于显示图片)
    需存在已训练好的模型文件 catdog_cnn.pt (运行过 train.py)

用法:
    python predict_ui.py
"""
import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog

import torch
from PIL import Image, ImageTk

from predict import CLASS_NAMES, load_model, make_transform, predict_image

THUMB_SIZE = 200      # 左侧缩略图边长(像素)
GRID_COLS = 3         # 左侧每行放几张缩略图
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# 置信度颜色: 高=绿, 中=橙, 低=红
COLOR_CONF = {"high": "#2e7d32", "mid": "#f9a825", "low": "#c62828", "none": "#999999"}
FONT = ("Microsoft YaHei", 10)          # 微软雅黑, Windows 中文显示友好


class ScrollableFrame(tk.Frame):
    """带竖向滚动条、支持鼠标滚轮的滚动容器"""

    def __init__(self, parent):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        vsb = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.inner = tk.Frame(self.canvas)
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        # 内容尺寸变化时更新滚动范围
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(
            scrollregion=self.canvas.bbox("all")))
        # 窗口宽度变化时让内容跟着变宽
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(
            self._win, width=e.width))

    def scroll_wheel(self, event):
        # Windows 上滚轮事件 delta 为 ±120 的倍数
        self.canvas.yview_scroll(int(-event.delta / 120), "units")


class PredictApp:
    def __init__(self, root):
        self.root = root
        root.title("猫狗识别结果展示")
        root.geometry("1150x720")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.queue = queue.Queue()          # 工作线程 → 主线程的消息队列
        self.worker_done = threading.Event()

        self._build_ui()
        # 全局滚轮: 鼠标悬停在哪个面板就滚动哪个
        root.bind_all("<MouseWheel>", self._on_wheel)

    # -------------------- 界面搭建 --------------------
    def _build_ui(self):
        top = tk.Frame(self.root)
        top.pack(side="top", fill="x", padx=8, pady=6)
        tk.Button(top, text="选择文件夹", command=self.choose_folder).pack(side="left")
        tk.Button(top, text="选择图片", command=self.choose_file).pack(side="left", padx=(6, 0))
        self.status = tk.Label(top, text="就绪", anchor="w", fg="#666")
        self.status.pack(side="left", padx=14)

        body = tk.Frame(self.root)
        body.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        tk.Label(body, text="图片预览", font=("Microsoft YaHei", 11, "bold")).grid(
            row=0, column=0, sticky="w")
        tk.Label(body, text="预测结果  (绿=有把握 橙=一般 红=接近猜)", font=FONT, fg="#888").grid(
            row=0, column=1, sticky="w", padx=(10, 0))

        self.left = ScrollableFrame(body)
        self.left.grid(row=1, column=0, sticky="nsew")
        self.right = ScrollableFrame(body)
        self.right.grid(row=1, column=1, sticky="nsew", padx=(10, 0))

        body.grid_rowconfigure(1, weight=1)
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)

    def _on_wheel(self, event):
        w = self.root.winfo_containing(event.x_root, event.y_root)
        while w is not None:
            if isinstance(w, ScrollableFrame):
                w.scroll_wheel(event)
                return
            w = w.master

    # -------------------- 选择图片/文件夹 --------------------
    def choose_folder(self):
        folder = filedialog.askdirectory(title="选择一个文件夹")
        if not folder:
            return
        files = sorted(os.path.join(folder, f) for f in os.listdir(folder)
                       if f.lower().endswith(IMG_EXTS))
        if not files:
            self.status.config(text="该文件夹下没有图片")
            return
        self.start_predict(files)

    def choose_file(self):
        path = filedialog.askopenfilename(
            title="选择一张图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self.start_predict([path])

    # -------------------- 预测流程 --------------------
    def start_predict(self, files):
        self.clear_panels()
        self.worker_done.clear()
        threading.Thread(target=self._worker, args=(files,), daemon=True).start()
        self.root.after(100, self._poll)

    def _worker(self, files):
        """后台线程: 逐张预测, 结果放进队列由主线程更新界面"""
        try:
            model = load_model(self.device)
        except SystemExit as e:
            self.queue.put(("error", str(e)))
            self.worker_done.set()
            return
        transform = make_transform()
        n_cat = n_dog = 0
        for i, path in enumerate(files, 1):
            try:
                label, conf = predict_image(model, transform, path, self.device)
                if label == "cat":
                    n_cat += 1
                elif label == "dog":
                    n_dog += 1
                self.queue.put(("done", i - 1, path, label, conf))
            except Exception:
                self.queue.put(("done", i - 1, path, None, 0.0))  # 无法读取
            self.queue.put(("progress", i, len(files)))
        self.queue.put(("summary", n_cat, n_dog))
        self.worker_done.set()

    def _poll(self):
        """主线程定时检查队列, 增量更新界面"""
        while True:
            try:
                item = self.queue.get_nowait()
            except queue.Empty:
                break
            kind = item[0]
            if kind == "error":
                self.status.config(text=item[1], fg="#c62828")
            elif kind == "done":
                _, idx, path, label, conf = item
                self.add_result(idx, path, label, conf)
            elif kind == "progress":
                self.status.config(text=f"预测中 {item[1]}/{item[2]} ...", fg="#666")
            elif kind == "summary":
                n_cat, n_dog = item[1], item[2]
                self.status.config(
                    text=f"完成: 共 {n_cat + n_dog} 张, 猫 {n_cat} 张, 狗 {n_dog} 张", fg="#2e7d32")
        if not (self.worker_done.is_set() and self.queue.empty()):
            self.root.after(100, self._poll)

    # -------------------- 结果展示 --------------------
    def add_result(self, idx, path, label, conf):
        name = os.path.basename(path)
        # 左侧: 缩略图
        col = idx % GRID_COLS
        row = idx // GRID_COLS
        cell = tk.Frame(self.left.inner)
        cell.grid(row=row, column=col, padx=6, pady=6)
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((THUMB_SIZE, THUMB_SIZE))
            photo = ImageTk.PhotoImage(img)
            lbl_img = tk.Label(cell, image=photo)
            lbl_img.image = photo  # 关键: 持有引用, 防止被垃圾回收后图片消失
            lbl_img.pack()
        except Exception:
            tk.Label(cell, text="[无法显示]", fg=COLOR_CONF["none"]).pack()
        tk.Label(cell, text=name, fg="#555", font=("Microsoft YaHei", 9)).pack()

        # 右侧: 预测结果行
        if label is None:
            text, color = "无法读取", COLOR_CONF["none"]
        else:
            cn = "猫" if label == "cat" else "狗"
            color = self._conf_color(conf)
            text = f"{cn}  {conf:.1%}"
        line = tk.Frame(self.right.inner)
        line.pack(anchor="w", fill="x", padx=10, pady=2)
        tk.Label(line, text=name, fg="#333", font=FONT, anchor="w").pack(side="left")
        tk.Label(line, text=text, fg=color, font=("Microsoft YaHei", 10, "bold"),
                 anchor="e").pack(side="right")

    def _conf_color(self, conf):
        if conf >= 0.8:
            return COLOR_CONF["high"]
        if conf >= 0.6:
            return COLOR_CONF["mid"]
        return COLOR_CONF["low"]

    def clear_panels(self):
        for child in self.left.inner.winfo_children():
            child.destroy()
        for child in self.right.inner.winfo_children():
            child.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    PredictApp(root)
    root.mainloop()
