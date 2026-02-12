import torch
import numpy as np
import fastapi
print(torch.__version__)
print(np.__version__)

print(torch.cuda.is_available())

# 打印当前 CUDA 版本
print(torch.version.cuda)

# 打印 CUDA 设备信息
print(torch.__config__.show())

