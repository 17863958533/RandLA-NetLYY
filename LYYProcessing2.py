# Path1 = '/media/pc/C8BE4D94BE4D7BC6/data/LYYData/LYYNewDataTXT/1MiddleCompositeSupportNew.txt'
# Path2 = '/media/pc/C8BE4D94BE4D7BC6/data/LYYData/LYYNewDataTXT/2MiddleSteelStructureNew.txt'
# Path3 = '/media/pc/C8BE4D94BE4D7BC6/data/LYYData/LYYNewDataTXT/3LandNew.txt'
# Path4 = '/media/pc/C8BE4D94BE4D7BC6/data/LYYData/LYYNewDataTXT/4TowerCraneNew.txt'
# Path5 = '/media/pc/C8BE4D94BE4D7BC6/data/LYYData/LYYNewDataTXT/5SupportFrameNew.txt'
# Path6 = '/media/pc/C8BE4D94BE4D7BC6/data/LYYData/LYYNewDataTXT/6ConstructionVehiclesNew.txt'
# Path7 = '/media/pc/C8BE4D94BE4D7BC6/data/LYYData/LYYNewDataTXT/7DomeNew.txt'
# Path8 = '/media/pc/C8BE4D94BE4D7BC6/data/LYYData/LYYNewDataTXT/8DomeBottomBracketNew.txt'
# Path0 = '/media/pc/C8BE4D94BE4D7BC6/data/LYYData/LYYNewDataTXT/0UnlabeledNew.txt'


Path1 = '/media/pc/C8BE4D94BE4D7BC6/data/LYYData3/newData/trainData1.txt'
Path2 = '/media/pc/C8BE4D94BE4D7BC6/data/LYYData3/newData/trainData2.txt'
Path3 = '/media/pc/C8BE4D94BE4D7BC6/data/LYYData3/newData/trainData3.txt'
Path4 = '/media/pc/C8BE4D94BE4D7BC6/data/LYYData3/newData/trainData4.txt'
Path5 = '/media/pc/C8BE4D94BE4D7BC6/data/LYYData3/newData/trainData5.txt'
Path6 = '/media/pc/C8BE4D94BE4D7BC6/data/LYYData3/newData/trainData6.txt'
Path7 = '/media/pc/C8BE4D94BE4D7BC6/data/LYYData3/newData/trainData7.txt'
Path8 = '/media/pc/C8BE4D94BE4D7BC6/data/LYYData3/newData/trainData8.txt'
Path9 = '/media/pc/C8BE4D94BE4D7BC6/data/LYYData3/newData/trainData9.txt'
Path10 = '/media/pc/C8BE4D94BE4D7BC6/data/LYYData3/newData/trainData10.txt'
Path11 = '/media/pc/C8BE4D94BE4D7BC6/data/LYYData3/newData/trainData11.txt'
Path12 = '/media/pc/C8BE4D94BE4D7BC6/data/LYYData3/newData/trainData12.txt'


file_names = [Path1, Path2, Path3, Path4, Path5, Path6, Path7, Path8, Path9, Path10, Path11, Path12]


#-------------------Change ply format data to txt format data.---------------------------
# # 定义输入PLY文件和输出TXT文件的路径
# input_ply_file = '/media/pc/C8BE4D94BE4D7BC6/data/LYYData2/original_ply/trainData1.ply'  # 输入PLY文件名
# output_txt_file = '/media/pc/C8BE4D94BE4D7BC6/data/LYYData2/linshi/trainData1.txt'  # 输出TXT文件名
#
# # 打开输入PLY文件以二进制读取模式
# with open(input_ply_file, 'rb') as infile:
#     # 打开输出TXT文件以写入模式
#     with open(output_txt_file, 'w') as outfile:
#         # 初始化一个标志，表示点云数据开始的地方
#         started = False
#
#         for line_bytes in infile:
#             # 将字节数据解码为字符串，默认使用UTF-8编码
#             try:
#                 line = line_bytes.decode('utf-8', errors='ignore').strip()
#             except UnicodeDecodeError:
#             # 如果解码失败，尝试使用ISO-8859-1编码
#                 line = line_bytes.decode('ISO-8859-1').strip()
#
#             # 查找以 "end_header" 开头的行，这表示点云数据的开始
#             if line.startswith("end_header"):
#                 started = True
#                 continue
#
#             # 如果已经开始了，就将点云数据写入输出TXT文件
#             if started:
#                 parts = line.strip().split()
#
#                 # 提取点的坐标和颜色信息（如果有）
#                 x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
#                 r, g, b = 0, 0, 0  # 如果没有颜色信息，可以自行设置默认颜色
#
#                 if len(parts) >= 6:
#                     r, g, b = int(parts[3]), int(parts[4]), int(parts[5])
#
#                 # 写入点的信息到TXT文件
#                 txt_line = f"{x} {y} {z} {r} {g} {b}\n"  # 最后的0表示属性标签，你可以根据需要自行更改
#                 outfile.write(txt_line)
#
# print(f'点云数据已转换为TXT文件，并保存在 {output_txt_file}')

#----------------------This is change the folat label numbers to int label numbers. And add sonme label numbers in every traindata and valdata, if not do this , the traing processing maybe come out wrong.--------------
# import os
# from tqdm import tqdm
#
# # 定义输入文件夹和输出文件夹的路径
# input_folder = '/media/pc/C8BE4D94BE4D7BC6/data/LYYData3/oldData/'  # 输入文件夹
# output_folder = '/media/pc/C8BE4D94BE4D7BC6/data/LYYData3/newData/'  # 输出文件夹
#
# # 创建输出文件夹
# os.makedirs(output_folder, exist_ok=True)
#
# # 创建一个字典，用于将浮点数标签映射为整数
# float_to_int_mapping = {0.0: 0, 1.0: 1, 2.0: 2, 3.0: 3, 4.0: 4, 5.0: 5}
#
# # 获取输入文件夹中的所有txt文件
# input_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
#
# # 添加的6行内容
# additional_lines = [
#     "143.32099915 8.96399498 20.13399696 212 212 0 8\n",
#     "137.75799561 7.98799467 20.13899612 212 212 0 8\n",
#     "140.19700623 8.39199448 20.49399757 212 212 0 8\n",
#     "138.35800171 10.31099510 20.36899757 212 212 0 8\n",
#     "140.91600037 11.76499462 20.22499657 212 212 0 8\n",
#     "148.25199890 9.86799526 20.36999702 212 212 0 8\n"
# ]
#
# for input_file in input_files:
#     input_file_path = os.path.join(input_folder, input_file)
#     output_file_path = os.path.join(output_folder, input_file)
#
#     # 打开输入文件以读取模式和输出文件以写入模式
#     with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
#         # 使用tqdm创建进度条
#         num_lines = sum(1 for line in infile)
#         infile.seek(0)  # 将文件指针重新定位到文件开头
#         progress_bar = tqdm(total=num_lines, desc=f'Processing {input_file}', unit='lines')
#
#         for line in infile:
#             # 分割行中的数据并将浮点数标签转换为整数
#             parts = line.strip().split()
#             x, y, z, r, g, b, l = map(float, parts)
#             l = float_to_int_mapping.get(l, l)
#
#             # 将整数数据重新格式化为字符串
#             formatted_line = f"{x} {y} {z} {r} {g} {b} {l:.0f}\n"
#
#             # 将格式化后的行写入输出文件
#             outfile.write(formatted_line)
#
#             # 更新进度条
#             progress_bar.update(1)
#
#         # 添加6行内容
#         #for additional_line in additional_lines:
#         #    outfile.write(additional_line)
#
#         # 关闭进度条
#         progress_bar.close()
#
# print(f'数据处理完成，结果保存在 {output_folder} 文件夹中。')

# #-------------This is used for counting every class point number ------------------------------------------
from tqdm import tqdm
#file_names = [Path1, Path2, Path3, Path4, Path5, Path6, Path7]
# 创建一个字典，用于存储每一类标签的点的个数
label_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

# 遍历每个文件
for file_name in file_names:
    with open(file_name, 'r') as file:
        num_lines = sum(1 for _ in file)
    with open(file_name, 'r') as file:
        for line in tqdm(file, total=num_lines, desc=f'处理文件 {file_name}'):
            parts = line.strip().split()
            if len(parts) >= 7:  # 确保每行至少有7个数据项
                label = float(parts[-1])  # 获取最后一个数字作为标签
                if label in label_counts:
                    label_counts[label] += 1

# 打印每一类标签的点的个数
for label, count in label_counts.items():
    print(f'标签 {label:.0f}: {count} 个点')








# #This is count the point numbers in every txt data file.------------------------
# for file_name in file_names:
#     line_count = 0
#     with open(file_name, 'r') as file:
#         for line in file:
#             line_count += 1
#     print(f'{file_name}: {line_count} lines')
#
# #This is output 1 txt by normal file index.-----------------------------------------
# output_file = '/media/pc/C8BE4D94BE4D7BC6/data/LYYData/AllData/allIndexSmall.txt'  # 保存合并结果的文件名
#
# # 打开输出文件以写入模式
# with open(output_file, 'w') as outfile:
#     for file_name in file_names:
#         print(file_name)
#         with open(file_name, 'r') as infile:
#             for line in infile:
#                 outfile.write(line)
#
# print(f'合并完成，结果保存在 {output_file}')

#This is output 1 txt by random file index.-------------------------------------------
# import random
#
# output_file = '/media/pc/C8BE4D94BE4D7BC6/data/LYYData/AllData/allIndexRandom_ValData.txt'  # 保存合并结果的文件名  # 保存打乱整合结果的文件名
#
# # 读取每个文件的行并放入一个列表中
# lines = []
# for file_name in file_names:
#     print(file_name)
#     with open(file_name, 'r') as infile:
#         lines.extend(infile.readlines())
#
# # 打乱行的顺序
# random.shuffle(lines)
#
# # 将打乱后的行写入输出文件
# with open(output_file, 'w') as outfile:
#     outfile.writelines(lines)
#
# print(f'打乱整合完成，结果保存在 {output_file}')
# #---------------------------------------------------------------------
# # 定义输入文件和输出文件的路径
# input_file = '/media/pc/C8BE4D94BE4D7BC6/data/LYYData/AllData/allIndexRandom_ValData.txt'  # 大的输入文件
# output_paths = ['/media/pc/C8BE4D94BE4D7BC6/data/LYYData/AllData/output_file1.txt',
#                 '/media/pc/C8BE4D94BE4D7BC6/data/LYYData/AllData/output_file2.txt',
#                 '/media/pc/C8BE4D94BE4D7BC6/data/LYYData/AllData/output_file3.txt']  # 3个输出文件
#
# # 定义每个输出文件的行数
# lines_per_file = []
#
# # 读取输入文件，统计总行数，并确定每个输出文件应包含的行数
# with open(input_file, 'r') as infile:
#     total_lines = sum(1 for line in infile)
#     lines_per_file = [total_lines // 3] * 3  # 尽可能平均地分成3个文件
#     # 如果总行数不能被3整除，将余数分配给前几个文件
#     for i in range(total_lines % 3):
#         lines_per_file[i] += 1
#
# # 分割输入文件为3个输出文件
# with open(input_file, 'r') as infile:
#     for output_path, num_lines in zip(output_paths, lines_per_file):
#         with open(output_path, 'w') as outfile:
#             for _ in range(num_lines):
#                 line = infile.readline()
#                 if line:
#                     outfile.write(line)
#
# print(f'文件成功分割为{len(output_paths)}个文件。')


#----------------------------------------------------------
# import numpy as np
#
# num_per_class = [5181602, 5012952, 6830086, 1311528, 10476365, 946982, 334860, 269353]
#
# # weight = num_per_class / float(sum(num_per_class))
# # weight2 = np.divide(num_per_class, float(sum(num_per_class)))
#
# total_num = np.sum(num_per_class)
# ce_label_weight3 = 1 / (num_per_class / total_num + 0.02)
#
# # ce_label_weight = 1 / (weight + 0.02)
# # ce_label_weight2 = 1 / (weight2 + 0.02)
#
# print(ce_label_weight3)
#----------------------------------------------------------
# # 定义输入文件和输出文件的路径
# input_file = '/media/pc/C8BE4D94BE4D7BC6/data/LYYData/LYYOldDataTXT/val_data2.txt'  # 输入文件名
# output_file = '/media/pc/C8BE4D94BE4D7BC6/data/LYYData/LYYNewDataTXT/val_data2New.txt'  # 输出文件名
#
# # 创建一个字典，用于将浮点数映射为整数
# float_to_int_mapping = {0.000000: 0, 1.000000: 1, 2.000000: 2, 3.000000: 3, 4.000000: 4, 5.000000: 5, 6.000000: 6, 7.000000: 7, 8.000000: 8}
#
# # 打开输入文件以读取模式和输出文件以写入模式
# with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
#     for line in infile:
#         # 分割行中的数据并将浮点数转换为整数
#         parts = line.strip().split()
#         x, y, z, r, g, b, l = map(float, parts)
#         l = float_to_int_mapping.get(l, l)
#
#         # 将整数数据重新格式化为字符串
#         formatted_line = f"{x} {y} {z} {r} {g} {b} {l:.0f}\n"
#
#         # 将格式化后的行写入输出文件
#         outfile.write(formatted_line)
#
# print(f'数据已经格式化并保存在 {output_file}')

