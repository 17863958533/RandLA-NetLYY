# Link: https://blog.csdn.net/u014311125/article/details/121226852
# This is convert the original data to .las format, so we can see these pointclouds in Cloudcompare.

import numpy as np
import laspy
import os

from scipy.constants import pt

txt_file = "/home/pc/myGitHub/RandLA-Net/DataSee/txtfile/testData3_noLable.txt"
label_file = "/home/pc/myGitHub/RandLA-Net/DataSee/labelfile/testData3.labels"
stack_file = "/home/pc/myGitHub/RandLA-Net/DataSee/stackfile/testData3_my.txt"

point_cloud = np.loadtxt(txt_file)
print(point_cloud)
label = np.loadtxt(label_file)
print(label.shape)

# 获取标签不为0的点mask
mask = (label != 0)

# ***************************保存为txt格式******************************

np.savetxt(stack_file,np.hstack([point_cloud[mask],label[mask].reshape(-1,1)]))

# 不去除标签为0的点，直接拼接后保存
#np.savetxt(stack_file,np.hstack([point_cloud,label.reshape(-1,1)]))

# ***************************保存为las格式******************************
# data_all = np.stack([pt[mask], label[mask].reshape(-1, 1)])
#
# las = laspy.create(file_version="1.2", point_format=3)
# las.x = data_all[:, 0]
# las.y = data_all[:, 1]
# las.z = data_all[:, 2]
# las.red = data_all[:, 4]
# las.green = data_all[:, 5]
# las.blue = data_all[:, 6]
#
# las.raw_classification = data_all[:, -1]
# las.write(stack_file)


