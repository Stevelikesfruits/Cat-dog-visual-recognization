import torch
print(torch.__version__)          # 应显示 2.x+cu126
print(torch.cuda.is_available())  # 应为 True
print(torch.cuda.get_device_name(0))  # 应显示 RTX 3060 Ti