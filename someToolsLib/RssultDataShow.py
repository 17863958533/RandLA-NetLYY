import numpy as np
import laspy
import os

ascii_files = {
    "MarketplaceFeldkirch_Station4_rgb_intensity-reduced.txt": "marketsquarefeldkirch4-reduced.labels",
    "StGallenCathedral_station6_rgb_intensity-reduced.txt": "stgallencathedral6-reduced.labels",
    "sg27_station10_rgb_intensity-reduced.txt": "sg27_10-reduced.labels",
    "sg28_Station2_rgb_intensity-reduced.txt": "sg28_2-reduced.labels"
}
# 原始点云文件
txt_file_dir = r"/home/pc/myGitHub/RandLA-Net/DataSee/txtfile/"
# 测试结果labels文件
test_label_dir = r"/home/pc/myGitHub/RandLA-Net/DataSee/labelfile/"
stack_file_dir = test_label_dir

for txt_file_name in ascii_files:
    txt_file = os.path.join(txt_file_dir, txt_file_name)
    test_label = os.path.join(test_label_dir, ascii_files[txt_file_name] + ".labels")
    stack_file = os.path.join(stack_file_dir, ascii_files[txt_file_name] + ".las")
    point_cloud = np.loadtxt(txt_file)
    print(point_cloud)
    label = np.loadtxt(label_file)
    print(label.shape)

    data_all = np.stack([pt, label.reshape(-1, 1)])

    las = laspy.create(file_version="1.2", point_format=3)
    las.x = data_all[:, 0]
    las.y = data_all[:, 1]
    las.z = data_all[:, 2]
    las.red = data_all[:, 4]
    las.green = data_all[:, 5]
    las.blue = data_all[:, 6]

    las.raw_classification = data_all[:, -1]
    las.write(las_file)
