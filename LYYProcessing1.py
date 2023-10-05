# 指定原始txt文件路径和新文件路径
input_file_path = '/media/pc/C8BE4D94BE4D7BC6/data/LYYData/LYYOldDataTXT/0Unlabeled.txt'
output_file_path = '/media/pc/C8BE4D94BE4D7BC6/data/LYYData/LYYNewDataTXT/0UnlabeledNew.txt'

# 指定要替换的标签类别整数N
replacement_label = 0

# 打开原始文件和新文件
with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
    # 逐行读取原始文件
    for line in input_file:
        # 将每一行按空格分割成单独的数据项
        items = line.strip().split()

        # 检查每行是否包含足够的数据项，如果不够，可以根据需要处理
        if len(items) < 7:
            continue

        # 获取X、Y、Z、R、G、B和L的值
        X, Y, Z, R, G, B, L = items[:7]

        # 将T删除（如果存在）
        # 将L替换为指定的整数N
        new_line = f'{X} {Y} {Z} {R} {G} {B} {replacement_label}\n'

        # 将处理后的行写入新文件
        output_file.write(new_line)

print(f'处理完成，结果已保存到{output_file_path}')
