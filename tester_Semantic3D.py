import os
from os import makedirs
from os.path import exists, join
from helper_ply import read_ply, write_ply
#import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import time
import laspy
from tqdm import tqdm

def log_string(out_str, log_out):
    log_out.write(out_str + '\n')
    log_out.flush()
    print(out_str)


class ModelTester:
    def __init__(self, model, dataset, restore_snap=None):
        # Tensorflow Saver definition
        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(my_vars, max_to_keep=100)

        # Create a session for running Ops on the Graph.
        on_cpu = False
        if on_cpu:
            c_proto = tf.ConfigProto(device_count={'GPU': 0})
        else:
            c_proto = tf.ConfigProto()
            c_proto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=c_proto)
        self.sess.run(tf.global_variables_initializer())

        if restore_snap is not None:
            self.saver.restore(self.sess, restore_snap)
            print("Model restored from " + restore_snap)

        # Add a softmax operation for predictions
        self.prob_logits = tf.nn.softmax(model.logits)
        # self.test_probs = [np.zeros((l.data.shape[0], model.config.num_classes), dtype=np.float16) for l in dataset.input_trees['test']]   #!!!

        # 初始化一个空的列表，用于存储 self.test_probs---------------------------------------
        self.test_probs = []
        # 遍历 input_trees 的内容
        for tree_data in dataset.input_trees:
            num_points = tree_data.data.shape[0]  # 获取每个内容中的点数

            # 创建一个与内容中点数对应的零矩阵，并添加到 self.test_probs 列表中
            zero_matrix = np.zeros((num_points, model.config.num_classes), dtype=np.float16)
            self.test_probs.append(zero_matrix)

        # self.test_probs = [np.zeros((dataset.input_trees[i].data.shape[0], model.config.num_classes), dtype=np.float16) for i in [0, 1, 2]]
        # -----------------------------------------------------------------------------
        self.log_out = open('log_test_' + dataset.name + '.txt', 'a')



    def test(self, model, dataset, test_path, num_votes=100):

        # 加权平均的平滑系统
        test_smooth = 0.98

        # 初始化测试集的dataset
        self.sess.run(dataset.test_init_op)

        #####################
        # Network predictions
        #####################

        step_id = 0
        epoch_id = 0
        last_min = -0.5

        # 逐轮测试
        while last_min < num_votes:

            try:
                # 操作列表，预测结果，标签，输入点云的id号，输入点云文件的id号
                ops = (self.prob_logits,
                       model.labels,
                       model.inputs['input_inds'],
                       model.inputs['cloud_inds'],)

                # 输出结果与输入一一对应
                stacked_probs, stacked_labels, point_idx, cloud_idx = self.sess.run(ops, {model.is_training: False})
                # 输出的预测结果重新reshape为[B,N,C]形式
                stacked_probs = np.reshape(stacked_probs, [model.config.val_batch_size, model.config.num_points,
                                                           model.config.num_classes])

                # 在一个batch_size中逐个的拼接输出的结果
                for j in range(np.shape(stacked_probs)[0]):
                    probs = stacked_probs[j, :, :]  # 第j个点云块的预测结果
                    inds = point_idx[j, :]  # 第j个点云块中点的id值
                    c_i = cloud_idx[j][0]  # 第j个点云块所在点云文件的id值

                    """
                    加权平均
                    将预测的结果按照一定的权重加到上一次结果中。
                    self.test_probs表示的是整个点云文件的预测结果
                    模型每次只预测一个点云块的结果，将这个点云块按照文件号，点号对应到整个点云预测结果上。
                    """
                    self.test_probs[c_i][inds] = test_smooth * self.test_probs[c_i][inds] + (1 - test_smooth) * probs
                step_id += 1
                # log_string('Epoch {:3d}, step {:3d}. min possibility = {:.1f}'.format(epoch_id, step_id, np.min(dataset.min_possibility['test'])), self.log_out)   #!!!

                # 遍历 min_possibility 中的值并找到最小值-----------------------------------
                for index, content in enumerate(dataset.min_possibility):
                    # 获取content中的最小值
                    min_values = np.min(content)
                log_string('Epoch {:3d}, step {:3d}. min possibility = {:.1f}'.format(epoch_id, step_id, np.min(min_values)), self.log_out)
                # ------------------------------------------------------------------------


            except tf.errors.OutOfRangeError:

                # 一轮完成后统计测试集中所有点的最小概率值
                # new_min = np.min(dataset.min_possibility['test'])    #!!!

                # 遍历 min_possibility 中的值并找到最小值-----------------------------------
                for index, content in enumerate(dataset.min_possibility):
                    # 获取content中的最小值
                    new_min = np.min(content)
                #------------------------------------------------------------------------

                log_string('Epoch {:3d}, end. Min possibility = {:.1f}'.format(epoch_id, new_min), self.log_out)

                if last_min + 4 < new_min:
                    # 如果最小概率值大于3.5，则测试结束，开始处理测试结果
                    print('Saving clouds')

                    # Update last_min
                    last_min = new_min

                    # Project predictions
                    print('\nReproject Vote #{:d}'.format(int(np.floor(new_min))))
                    t1 = time.time()
                    files = dataset.test_file  # 测试文件
                    remove_outlier_folder = dataset.remove_outlier_pc_folder
                    original_folder = dataset.original_folder
                    #---------------------------
                    # TestPath1 = '/media/pc/C8BE4D94BE4D7BC6/data/LYYData4/LYYinput/input_0.060/testData1.ply'
                    # TestPath2 = '/media/pc/C8BE4D94BE4D7BC6/data/LYYData4/LYYinput/input_0.060/testData2.ply'
                    # TestPath3 = '/media/pc/C8BE4D94BE4D7BC6/data/LYYData4/LYYinput/input_0.060/testData3.ply'
                    # files = [TestPath1, TestPath2, TestPath3]
                    # remove_outlier_folder = dataset.remove_outlier_pc_folder
                    # original_folder = '/media/pc/C8BE4D94BE4D7BC6/data/LYYData4/LYYinput/original_data/'
                    #----------------------------
                    i_test = 0
                    for i, file_path in enumerate(files):
                        # 读取测试文件的点云数据
                        points = self.load_evaluation_points(file_path)
                        points = points.astype(np.float16)
                        # print('chenggongduqu!!!')

                        # Reproject probs
                        probs = np.zeros(shape=[np.shape(points)[0], 8], dtype=np.float16)
                        proj_index = dataset.test_proj[i_test]

                        probs = self.test_probs[i_test][proj_index, :]

                        # Insert false columns for ignored labels
                        probs2 = probs
                        for l_ind, label_value in enumerate(dataset.label_values):
                            if label_value in dataset.ignored_labels:
                                probs2 = np.insert(probs2, l_ind, 0, axis=1)

                        # 使用argmax()得到预测标签
                        preds = dataset.label_values[np.argmax(probs2, axis=1)].astype(np.uint8)
                        print('Label predection finished!!!')

                        cloud_name = os.path.basename(file_path)

                        original_file = join(original_folder, cloud_name.replace(".ply", ".txt"))
                        print('The original data has been opened: ' + original_file)
                        # seg_result = join(test_path, cloud_name.replace(".ply", ".las"))
                        seg_result = join(test_path, cloud_name.replace(".ply", ".txt"))
                        print('The results are writing: ' + seg_result)
                        pcd = np.loadtxt(original_file)


                        if dataset.remove_outlier:
                            remove_outlier_id_file = join(remove_outlier_folder, cloud_name.replace(".ply", ".txt"))
                            nu_outlier_id = np.loadtxt(remove_outlier_id_file, dtype=np.longlong)
                            points_num = pcd.shape[0]
                            new_seg = np.ones(points_num, dtype=np.uint8) * (dataset.num_classes + 1)
                            new_seg[nu_outlier_id] = preds[nu_outlier_id]

                        # # 保存为las格式
                        # las = laspy.create(file_version="1.2", point_format=3)
                        # las.x = pcd[:, 0]
                        # las.y = pcd[:, 1]
                        # las.z = pcd[:, 2]
                        # if dataset.remove_outlier:
                        #     las.raw_classification = new_seg
                        # else:
                        #     las.raw_classification = preds
                        #
                        # las.write(seg_result)

                        #----------------------------！！！
                        X = pcd[:, 0]
                        Y = pcd[:, 1]
                        Z = pcd[:, 2]
                        if dataset.remove_outlier:
                            L = new_seg
                        else:
                            L = preds
                        # 指定输出txt文件的路径
                        output_txt_file = seg_result  # 替换成你的文件路径
                        # 获取矩阵的行数
                        num_rows = len(X)
                        # 使用进度条将数据写入txt文件
                        with open(output_txt_file, 'w') as txt_file:
                            for i in tqdm(range(num_rows), desc='Writing to txt', unit=' lines'):
                                x = X[i]
                                y = Y[i]
                                z = Z[i]
                                l = L[i]
                                txt_line = f"{x} {y} {z} {l}\n"
                                txt_file.write(txt_line)
                        print(f'Data has been written to {output_txt_file}')
                        #----------------------------

                        log_string(seg_result + 'has saved', self.log_out)
                        i_test += 1

                    t2 = time.time()
                    print('Done in {:.1f} s\n'.format(t2 - t1))
                    self.sess.close()
                    return

                # 如果测试没有结束则继续，先重新初始化测试集
                self.sess.run(dataset.test_init_op)
                epoch_id += 1
                step_id = 0
                continue
        return

    @staticmethod
    def load_evaluation_points(file_path):
        data = read_ply(file_path)
        return np.vstack((data['x'], data['y'], data['z'])).T
