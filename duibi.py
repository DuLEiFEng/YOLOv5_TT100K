"""
标题：
作者：DuLei
日期：2022年11月09日
"""
import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy  as np
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
# df1 = pd.read_csv("runs/train/yolov5/results.csv")   #读取文件1
df2 = pd.read_csv("train/aug0.001/results.csv")   #读取文件2
df3 = pd.read_csv("train/YOLOv5_Baseline_aug/results.csv")  #读取文件3
# df4 = pd.read_csv("runs/train/yolov5-spd/results.csv")   #读取文件4

# epoch_1 = df1["               epoch"].values.tolist()                     #通过文件表头信息读取文件内容
# mAP5_1  = df1["     metrics/mAP_0.5"].values.tolist()

epoch_2 = df2["               epoch"].values.tolist()                     #通过文件表头信息读取文件内容
mAP5_2  = df2["     metrics/mAP_0.5"].values.tolist()

epoch_3 = df3["               epoch"].values.tolist()                     #通过文件表头信息读取文件内容
mAP5_3  = df3["     metrics/mAP_0.5"].values.tolist()
# epoch_4 = df4["               epoch"].values.tolist()                     #通过文件表头信息读取文件内容
# mAP5_4  = df4["     metrics/mAP_0.5"].values.tolist()

plt.figure(figsize=(8, 5))
# plt.plot(epoch_1,mAP5_1,color='red',  label='yolov5s')       #设置曲线相关系数
plt.plot(epoch_2,mAP5_2,color='black',label='yolov5s_aug')       #设置曲线相关系数
plt.plot(epoch_3,mAP5_3,color='blue',label='yolov5s_C3ECA')       #设置曲线相关系数
# plt.plot(epoch_4,mAP5_4,color='green',label='yolov5s_tph')       #设置曲线相关系数


plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plt.ylim(0, 1)
plt.xlim(0, 400)                        #设置坐标轴取值范围
plt.xlabel('epochs', fontsize=14)
plt.ylabel('mAP_0.5', fontsize=14)
plt.legend(fontsize=12,loc="best") #设置标签位置及大小
plt.savefig("test.png",bbox_inches='tight')
plt.show()
