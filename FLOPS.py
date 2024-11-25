# -- coding: utf-8 --
import torch
import torchvision
from thop import profile
import model
import model1
# from model import *
import time
import numpy as np
import yaml
from new_mamba import *

import torch
import time
from thop import profile, clever_format

# 定义模型和输入张量
net = ULNet(num_classes=1, 
                               input_channels=3, 
                            #    c_list=[16,32,64,128,256,256], 
                            c_list=[8,16,32,64,128,128],
                               split_att='fc', 
                               bridge=False,)
# net = model1.DRNet()
# 移动模型到GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = net.to(device)

dummy_input = torch.randn(1, 3, 224, 224).to(device)

# 计算FLOPs和参数量
flops, params = profile(net, (dummy_input,))

# 将FLOPs转换为GFLOPs，将参数量转换为百万（M）
flops_in_gflops = flops*2 / 1e9
params_in_m = params / 1e6

# 格式化FLOPs和参数量
flops_str, params_str = clever_format([flops, params], "%.3f")

print(f"FLOPs: {flops_in_gflops:.3f} GFLOPs")
print(f"Params: {params_in_m:.3f} M")

# 计算FPS
net.eval()  # 设置模型为评估模式

# 预热GPU（可选，但推荐用于准确计时）
for _ in range(10):
    with torch.no_grad():
        net(dummy_input)

# 测量前向传播时间
num_iterations = 100
start_time = time.time()
with torch.no_grad():
    for _ in range(num_iterations):
        net(dummy_input)
end_time = time.time()

# 计算FPS
total_time = end_time - start_time
fps = num_iterations / total_time

print(f"FPS: {fps:.2f}")