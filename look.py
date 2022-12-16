"""
标题：
作者：DuLei
日期：2022年11月07日
"""
# #################查看模型 的 anchor  #######################
import torch
from models.experimental import attempt_load

model = attempt_load('runs/train/yolov5-tph/weights/best.pt', map_location=torch.device('cpu'))
m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]
print(m.anchor_grid)

