###这个是LYY部署完成的RandLA-Net网络，可以实现对模型的训练和结果的最终直接输出。

（1）首先请部署环境，直接参考tf26lyy.yaml文件即可，这里我对源程序进行了修改，使之可以适合tensorflow2.6环境，并且支持python3.6，同时在3070Ti显卡和Z790电脑主板上进行了测试，证明了新环境架构的可行性。使用命令：conda env create -f tf26lyy.yaml即可在anconda环境中直接部署和作者完全一致的开发环境。
请注意，使用前请先修改yaml文件中最后一行的环境路径，并确定环境的名称是您想要的。

（2）准备数据源，可以参考 数据集制作过程.txt文件里面的说明。


（3）设置好训练参数并开始训练，在此之前请检查你的电脑显卡的显存和内存的大小是否能够满足要求。推荐显存不低于10G，内存不低于32G。


（4）进行数据结果的测试。



需要说明的事项：

（1）utils_backup.zip压缩包中的文件是用来对数据处理模块的备份，以用来防止数据准备模块编译的有问题。

（2）训练数据的时候请确保电脑至少有32G的内存和10G的独立显存。

（3）数据制作的格式需要严格按照要求进行，否则无法训练。

（4）经过LYY的修改，data_prepare_LYYData.py脚本是用来处理标记数据的，生成可以用于训练的数据集，其中请将grid_size设置为合适的大小，且需要和helper_tool.py文件中设置的保持完全一样，指定的路径下放上txt格式的文件，其中每一行的格式为x y z r g b l，每一个数据用空格分割，每一行代表一个点，其中l表示类别，需要和主函数main中的类别序号保持完全一样，不能出现新的类别序号。

（5）在制作数据集的时候，需要划分三类txt文件，每一类txt文件的数量可以不一样，这三类数据集分别是trainData,valData,testData；其中trainData,valData的格式为x y z r g b l，其中testData的格式为x y z r g b，这些数据txt都是一次性的全部放到data_prepare_LYYData.py脚本指定的original_data文件夹中。

（6）整个RandLA-NetLYY项目工程中的脚本可以分成两组，第一组是：main_LYYData.py、data_prepare_LYYData.py（在utils目录下）、tester_LYYData.py、RandLANet.py（公用）、helper_tool.py（公用），这5个脚本配合可以开始对数据进行训练并在result文件夹下得到最终的模型； 第二组是：main_Semantic3D.py、tester_Semantic3D.py、RandLANet.py（公用）、helper_tool.py（公用），这4个脚本配合可以实现输入待检测的txt和训练好的模型，输出预测的txt结果，直接可以在CC里面进行查看。

（7）在运行任何一个main主程序之前，务必检查数据路径的正确，并检查helper_tool.py脚本中mydata类的带叹号的参数，尤其注意类别数量num_classes和sub_grid_size要与主函数中的设置相一致。

（8）如果你使用的是第一组程序组合进行训练，你可以通过PreData2Las.py脚本获得附加标签的最终检测出来的txt,需要注意的是这里是给采样后在original_ply文件夹中的.ply文件进行label标签的附加，加入其最后一列，视之能够通过CC进行结果的显示。

（9）grid_size设置的越小，表示采样后保留下来的点越多，和原点云越相似，计算量越大；num_points = 95536这个数值越大，表示搜索的范围越大，训练出的模型对宏观把控越好，计算成本越高。

![a](https://github.com/17863958533/RandLA-NetLYY/blob/main/imgs/%E9%80%89%E5%8C%BA_002.jpg)

![a](https://github.com/17863958533/RandLA-NetLYY/blob/main/imgs/%E9%80%89%E5%8C%BA_003.jpg)

![a](https://github.com/17863958533/RandLA-NetLYY/blob/main/imgs/%E9%80%89%E5%8C%BA_004.jpg)



相比与最原始版本的RandLA-Net网络，修改的部分如下：（该部分格式可能较乱。可以直接参考程序文件）


-------------修改main_Semantic3D.py------------------------------------------------------------------------------------------



from os.path import join, exists,dirname,abspath
from RandLANet import Network
from tester_Semantic3D import ModelTester
from helper_ply import read_ply,write_ply
from helper_tool import Plot
from helper_tool import DataProcessing as DP
from helper_tool import ConfigSemantic3D as cfg
import tensorflow as tf
import numpy as np
import pickle, argparse, os

import open3d as o3d

from sklearn.neighbors import KDTree
import glob
import sys

BASE_DIR=dirname(abspath(__file__))
ROOT_DIR=dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)


def preprocess_pc(dataset_path,remove_outlier=False):
    grid_size = 0.5


    sub_pc_folder = join(dirname(dataset_path), 'input_{:.3f}'.format(grid_size))
    os.mkdir(sub_pc_folder) if not exists(sub_pc_folder) else None

    for pc_path in glob.glob(join(dataset_path, '*.txt')):
        print(pc_path)
        # file_name = pc_path.split('/')[-1][:-4]
        file_name = os.path.basename(pc_path)[:-4]

        # check if it has already calculated
        if exists(join(sub_pc_folder, file_name + '_KDTree.pkl')):
            continue

        # 使用open3d加载点云，为了后面去除离群点。
        pcd=o3d.io.read_point_cloud(pc_path,format='xyzrgb')

        if remove_outlier:
            remove_outlier_folder = join(dirname(dataset_path), 'remove_outlier')
            os.makedirs(remove_outlier_folder)

            remove_outlier_inds=join(remove_outlier_folder,file_name+".txt")
            #去除离群点
            cd,ind=pcd.remove_radius_outlier(nb_points=16,radius=10)
            inlier_cloud=pcd.select_by_index(ind)
            np.savetxt(remove_outlier_inds,ind,fmt='%d')
            pc=np.array(inlier_cloud.ponits)

        else:
            pc=np.array(pcd.points)

        # 我的数据集中训练和测试集都是x,y,z,r,g,b,label的存储格式
        labels = pc[:, -1].astype(np.uint8)

        # save sub_cloud and KDTree file
        sub_xyz, sub_colors, sub_labels = DP.grid_sub_sampling(pc[:, :3].astype(np.float32),
                                                              pc[:, 3:6].astype(np.uint8), labels, grid_size)
        sub_colors = sub_colors / 255.0
        sub_labels = np.squeeze(sub_labels)
        sub_ply_file = join(sub_pc_folder, file_name + '.ply')
        write_ply(sub_ply_file, [sub_xyz, sub_colors, sub_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

        search_tree = KDTree(sub_xyz, leaf_size=50)
        kd_tree_file = join(sub_pc_folder, file_name + '_KDTree.pkl')
        with open(kd_tree_file, 'wb') as f:
            pickle.dump(search_tree, f)

        proj_idx = np.squeeze(search_tree.query(pcd.points, return_distance=False))
        proj_idx = proj_idx.astype(np.int32)
        proj_save = join(sub_pc_folder, file_name + '_proj.pkl')
        with open(proj_save, 'wb') as f:
            pickle.dump([proj_idx, labels], f)
	    
            
#-----------------修改main_Semantic3D.py---------------------------------------------------------------------------------



class Power:
    def __init__(self,path,remove_outlier=False):
        self.name = 'power'
        self.path = path
        self.label_to_names = {0: 'background',
                               1: 'tower',
                               2: 'line',
                               3: 'building',
                               4: 'vegetation'
                               }
        self.remove_outlier=remove_outlier
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.sort([])

        self.original_folder = join(self.path, 'original_data')
        self.remove_outlier_pc_folder = join(self.path, 'remove_outlier')
        self.sub_pc_folder = join(self.path, 'input_{:.3f}'.format(cfg.sub_grid_size))


        # Initial training-validation-testing files
        self.files = []
        cloud_names = [file_name[:-4] for file_name in os.listdir(self.original_folder) if file_name[-4:] == '.txt']

        #根据文件名加载数据
        for pc_name in cloud_names:
            pc_file=join(self.sub_pc_folder, pc_name + '.ply')
            self.files.append(pc_file)


        # Initiate containers
        self.test_proj = []
        self.test_labels = []

        self.possibility = {}
        self.min_possibility = {}
        self.class_weight = {}

        self.input_trees = []
        self.input_colors = []
        self.input_labels = []

        self.load_sub_sampled_clouds(cfg.sub_grid_size)

    def load_sub_sampled_clouds(self, sub_grid_size):

        tree_path = join(self.path, 'input_{:.3f}'.format(sub_grid_size))

        for i, file_path in enumerate(self.files):
            cloud_name = file_path.split('/')[-1][:-4]
            print('Load_pc_' + str(i) + ': ' + cloud_name)

            # Name of the input files
            kd_tree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))

            # read ply with data
            data = read_ply(sub_ply_file)
            sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T


            # Read pkl with search tree
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            self.input_trees += [search_tree]
            self.input_colors += [sub_colors]

        # Get validation and test re_projection indices
        print('\nPreparing reprojection indices for validation and test')

        for i, file_path in enumerate(self.files):

            # get cloud name and split
            cloud_name = file_path.split('/')[-1][:-4]


            # Test projection
            proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
            with open(proj_file, 'rb') as f:
                proj_idx, labels = pickle.load(f)
            self.test_proj += [proj_idx]
            self.test_labels += [labels]
        print('finished')
        return

    # Generate the input data flow
    def get_batch_gen(self):

        num_per_epoch = cfg.val_steps * cfg.val_batch_size

        # Reset possibility
        self.possibility = []
        self.min_possibility = []
        self.class_weight = []

        # Random initialize
        for i, tree in enumerate(self.input_trees):
            self.possibility += [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_possibility += [float(np.min(self.possibility[-1]))]

        def spatially_regular_gen():

            # Generator loop
            for i in range(num_per_epoch):  # num_per_epoch

                # Choose the cloud with the lowest probability
                cloud_idx = int(np.argmin(self.min_possibility))

                # choose the point with the minimum of possibility in the cloud as query point
                point_ind = np.argmin(self.possibility[cloud_idx])

                # Get all points within the cloud from tree structure
                points = np.array(self.input_trees[cloud_idx].data, copy=False)

                # Center point of input region
                center_point = points[point_ind, :].reshape(1, -1)

                # Add noise to the center point
                noise = np.random.normal(scale=cfg.noise_init / 10, size=center_point.shape)
                pick_point = center_point + noise.astype(center_point.dtype)
                query_idx = self.input_trees[cloud_idx].query(pick_point, k=cfg.num_points)[1][0]

                # Shuffle index
                query_idx = DP.shuffle_idx(query_idx)

                #根据序号获取周围点的坐标
                queried_pc_xyz = points[query_idx]
                #将获取到的点的x，y归一化为0
                queried_pc_xyz[:, 0:2] = queried_pc_xyz[:, 0:2] - pick_point[:, 0:2]
                #获取颜色
                queried_pc_colors = self.input_colors[cloud_idx][query_idx]


                queried_pc_labels = np.zeros(queried_pc_xyz.shape[0])
                queried_pt_weight = 1


                # 计算邻域点相对于中心点的欧式距离
                dists = np.sum(np.square((points[query_idx] - pick_point).astype(np.float32)), axis=1)
                """
                    使用距离和权重计算邻域点的概率值，距离中心点越近的，点数越多的类别概率增加越大，
                    这样可以防止一个点被多次选中，也可以防止类别多的点被多次选中，而类别少的点很少被选中。
                """
                delta = np.square(1 - dists / np.max(dists)) * queried_pt_weight
                #更新概率值
                self.possibility[cloud_idx][query_idx] += delta
                #重新计算最小概率值
                self.min_possibility[cloud_idx] = float(np.min(self.possibility[cloud_idx]))

                if True:
                    #通过yield返回数据
                    yield (queried_pc_xyz,   # 坐标
                           queried_pc_colors.astype(np.float32), #颜色
                           queried_pc_labels, #标签
                           query_idx.astype(np.int32), #点云id
                           np.array([cloud_idx], dtype=np.int32)) #点云所在的文件id

        gen_func = spatially_regular_gen
        gen_types = (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32)
        gen_shapes = ([None, 3], [None, 3], [None], [None], [None])
        return gen_func, gen_types, gen_shapes

    def get_tf_mapping(self):
    	“”“
    	不做修改
    	”“”

    # data augmentation
    @staticmethod
    def tf_augment_input(inputs):
    	“”
		不做修改
		“”

    def init_input_pipeline(self):
        print('Initiating input pipelines')
        cfg.ignored_label_inds = [self.label_to_idx[ign_label] for ign_label in self.ignored_labels]
        gen_function_test, gen_types, gen_shapes = self.get_batch_gen()
        self.test_data = tf.data.Dataset.from_generator(gen_function_test, gen_types, gen_shapes)
        self.batch_test_data = self.test_data.batch(cfg.val_batch_size)
        map_func = self.get_tf_mapping()
        self.batch_test_data = self.batch_test_data.map(map_func=map_func)
        self.batch_test_data = self.batch_test_data.prefetch(cfg.val_batch_size)

        iter = tf.data.Iterator.from_structure(self.batch_test_data.output_types, self.batch_test_data.output_shapes)
        self.flat_inputs = iter.get_next()
        self.test_init_op = iter.make_initializer(self.batch_test_data)

 

#--------------------------修改tester_Semantic3D.py-----------------------------------------------------------------


import os
from os import makedirs
from os.path import exists, join
from helper_ply import read_ply, write_ply
import tensorflow as tf
import numpy as np
import time
import laspy


def log_string(out_str, log_out):
    log_out.write(out_str + '\n')
    log_out.flush()
    print(out_str)


class ModelTester:
    def __init__(self, model, dataset, restore_snap=None):
        # Tensorflow Saver definition
        """
		不做修改
		"""

    def test(self, model, dataset,test_path,num_votes=100):

        # 加权平均的平滑系统
        test_smooth = 0.98

        #初始化测试集的dataset
        self.sess.run(dataset.test_init_op)

        #####################
        # Network predictions
        #####################

        step_id = 0
        epoch_id = 0
        last_min = -0.5

        #逐轮测试
        while last_min < num_votes:

            try:
                #操作列表，预测结果，标签，输入点云的id号，输入点云文件的id号
                ops = (self.prob_logits,
                       model.labels,
                       model.inputs['input_inds'],
                       model.inputs['cloud_inds'],)

                #输出结果与输入一一对应
                stacked_probs, stacked_labels, point_idx, cloud_idx = self.sess.run(ops, {model.is_training: False})
                #输出的预测结果重新reshape为[B,N,C]形式
                stacked_probs = np.reshape(stacked_probs, [model.config.val_batch_size, model.config.num_points,
                                                           model.config.num_classes])

                #在一个batch_size中逐个的拼接输出的结果
                for j in range(np.shape(stacked_probs)[0]):
                    probs = stacked_probs[j, :, :]  #第j个点云块的预测结果
                    inds = point_idx[j, :]          #第j个点云块中点的id值
                    c_i = cloud_idx[j][0]           #第j个点云块所在点云文件的id值

                    """
                    加权平均
                    将预测的结果按照一定的权重加到上一次结果中。
                    self.test_probs表示的是整个点云文件的预测结果
                    模型每次只预测一个点云块的结果，将这个点云块按照文件号，点号对应到整个点云预测结果上。
                    """
                    self.test_probs[c_i][inds] = test_smooth * self.test_probs[c_i][inds] + (1 - test_smooth) * probs
                step_id += 1
                log_string('Epoch {:3d}, step {:3d}. min possibility = {:.1f}'.format(epoch_id, step_id, np.min(
                    dataset.min_possibility['test'])), self.log_out)

            except tf.errors.OutOfRangeError:

                #一轮完成后统计测试集中所有点的最小概率值
                new_min = np.min(dataset.min_possibility['test'])
                log_string('Epoch {:3d}, end. Min possibility = {:.1f}'.format(epoch_id, new_min), self.log_out)


                if last_min + 4 < new_min:
                    #如果最小概率值大于3.5，则测试结束，开始处理测试结果
                    print('Saving clouds')

                    # Update last_min
                    last_min = new_min

                    # Project predictions
                    print('\nReproject Vote #{:d}'.format(int(np.floor(new_min))))
                    t1 = time.time()
                    files = dataset.test_files  #测试文件
                    remove_outlier_folder=dataset.remove_outlier_pc_folder
                    original_folder=dataset.origial_folder
                    i_test = 0
                    for i, file_path in enumerate(files):
                        # 读取测试文件的点云数据
                        points = self.load_evaluation_points(file_path)
                        points = points.astype(np.float16)

                        # Reproject probs
                        probs = np.zeros(shape=[np.shape(points)[0], 8], dtype=np.float16)
                        proj_index = dataset.test_proj[i_test]

                        probs = self.test_probs[i_test][proj_index, :]

                        # Insert false columns for ignored labels
                        probs2 = probs
                        for l_ind, label_value in enumerate(dataset.label_values):
                            if label_value in dataset.ignored_labels:
                                probs2 = np.insert(probs2, l_ind, 0, axis=1)

                        #使用argmax()得到预测标签
                        preds = dataset.label_values[np.argmax(probs2, axis=1)].astype(np.uint8)


                        cloud_name = os.path.basename(file_path)

                        original_file=join(original_folder,cloud_name.replace(".ply",".txt"))
                        seg_result=join(test_path,cloud_name.replace(".ply",".las"))
                        pcd=np.loadtxt(original_file)

                        if dataset.remove_outlier:
                            remove_outlier_id_file=join(remove_outlier_folder,cloud_name.replace(".ply",".txt"))
                            nu_outlier_id=np.loadtxt(remove_outlier_id_file,dtype=np.longlong)
                            points_num=pcd.shape[0]
                            new_seg=np.ones(points_num,dtype=np.uint8)*(dataset.num_classes+1)
                            new_seg[nu_outlier_id]=preds[nu_outlier_id]


                        #保存为las格式
                        las=laspy.create(file_version="1.2",point_format=3)
                        las.x=pcd[:,0]
                        las.y=pcd[:,1]
                        las.z=pcd[:,2]
                        if dataset.remove_outlier:
                            las.raw_classification=new_seg
                        else:
                            las.raw_classification=preds

                        las.write(seg_result)


                        log_string(seg_result + 'has saved', self.log_out)
                        i_test += 1

                    t2 = time.time()
                    print('Done in {:.1f} s\n'.format(t2 - t1))
                    self.sess.close()
                    return

                #如果测试没有结束则继续，先重新初始化测试集
                self.sess.run(dataset.test_init_op)
                epoch_id += 1
                step_id = 0
                continue
        return

    @staticmethod
    def load_evaluation_points(file_path):
        data = read_ply(file_path)
        return np.vstack((data['x'], data['y'], data['z'])).T
	


#----------------------------修改main_Semantic3D.py------------------------------------------------------------------------------------



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',type=str,default='None')
    parser.add_argument('--output_dir',type=str,default='None')
    parser.add_argument('--gpu', type=int, default=0, help='the number of GPUs to use [default: 0]')
    parser.add_argument('--remove_outlier', type=bool, default=False, help='options: True, False')
    parser.add_argument('--model_path', type=str, default='None', help='pretrained model path')
    FLAGS = parser.parse_args()

    GPU_ID = FLAGS.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
	
	origin_data=os.path.join(FLAGS.input_dir,"original_data")
	
	preprocess_pc(origin_data,FLAGS.remove_outlier)
	
    Mode = FLAGS.mode
    dataset = Power(FLAGS.input_dir,FLAGS.remove_outlier)
    dataset.init_input_pipeline()


    cfg.saving = False
    #网络初始化
    model = Network(dataset, cfg)
    tester = ModelTester(model, dataset, restore_snap=FLAGS.model_path)
    #开始测试
    tester.test(model, dataset,FLAGS.output_dir)



###------------------------------------------
下面致敬原作者！！！
###------------------------------------------

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/191111236/semantic-segmentation-on-semantic3d)](https://paperswithcode.com/sota/semantic-segmentation-on-semantic3d?p=191111236)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/191111236/3d-semantic-segmentation-on-semantickitti)](https://paperswithcode.com/sota/3d-semantic-segmentation-on-semantickitti?p=191111236)
[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)

# RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds (CVPR 2020)

This is the official implementation of **RandLA-Net** (CVPR2020, Oral presentation), a simple and efficient neural architecture for semantic segmentation of large-scale 3D point clouds. For technical details, please refer to:
 
**RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds** <br />
[Qingyong Hu](https://www.cs.ox.ac.uk/people/qingyong.hu/), [Bo Yang*](https://yang7879.github.io/), [Linhai Xie](https://www.cs.ox.ac.uk/people/linhai.xie/), [Stefano Rosa](https://www.cs.ox.ac.uk/people/stefano.rosa/), [Yulan Guo](http://yulanguo.me/), [Zhihua Wang](https://www.cs.ox.ac.uk/people/zhihua.wang/), [Niki Trigoni](https://www.cs.ox.ac.uk/people/niki.trigoni/), [Andrew Markham](https://www.cs.ox.ac.uk/people/andrew.markham/). <br />
**[[Paper](https://arxiv.org/abs/1911.11236)] [[Video](https://youtu.be/Ar3eY_lwzMk)] [[Blog](https://zhuanlan.zhihu.com/p/105433460)] [[Project page](http://randla-net.cs.ox.ac.uk/)]** <br />
 
 
<p align="center"> <img src="http://randla-net.cs.ox.ac.uk/imgs/Fig3.png" width="100%"> </p>


	
### (1) Setup
This code has been tested with Python 3.5, Tensorflow 1.11, CUDA 9.0 and cuDNN 7.4.1 on Ubuntu 16.04.
 
- Clone the repository 
```
git clone --depth=1 https://github.com/QingyongHu/RandLA-Net && cd RandLA-Net
```
- Setup python environment
```
conda create -n randlanet python=3.5
source activate randlanet
pip install -r helper_requirements.txt
sh compile_op.sh
```

**Update 03/21/2020, pre-trained models and results are available now.** 
You can download the pre-trained models and results [here](https://drive.google.com/open?id=1iU8yviO3TP87-IexBXsu13g6NklwEkXB).
Note that, please specify the model path in the main function (e.g., `main_S3DIS.py`) if you want to use the pre-trained model and have a quick try of our RandLA-Net.

### (2) S3DIS
S3DIS dataset can be found 
<a href="https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1">here</a>. 
Download the files named "Stanford3dDataset_v1.2_Aligned_Version.zip". Uncompress the folder and move it to 
`/data/S3DIS`.

- Preparing the dataset:
```
python utils/data_prepare_s3dis.py
```
- Start 6-fold cross validation:
```
sh jobs_6_fold_cv_s3dis.sh
```
- Move all the generated results (*.ply) in `/test` folder to `/data/S3DIS/results`, calculate the final mean IoU results:
```
python utils/6_fold_cv.py
```

Quantitative results of different approaches on S3DIS dataset (6-fold cross-validation):

![a](http://randla-net.cs.ox.ac.uk/imgs/S3DIS_table.png)

Qualitative results of our RandLA-Net:

| ![2](imgs/S3DIS_area2.gif)   | ![z](imgs/S3DIS_area3.gif) |
| ------------------------------ | ---------------------------- |



### (3) Semantic3D
7zip is required to uncompress the raw data in this dataset, to install p7zip:
```
sudo apt-get install p7zip-full
```
- Download and extract the dataset. First, please specify the path of the dataset by changing the `BASE_DIR` in "download_semantic3d.sh"    
```
sh utils/download_semantic3d.sh
```
- Preparing the dataset:
```
python utils/data_prepare_semantic3d.py
```
- Start training:
```
python main_Semantic3D.py --mode train --gpu 0
```
- Evaluation:
```
python main_Semantic3D.py --mode test --gpu 0
```
Quantitative results of different approaches on Semantic3D (reduced-8):

![a](http://randla-net.cs.ox.ac.uk/imgs/Semantic3D_table.png)

Qualitative results of our RandLA-Net:

| ![z](imgs/Semantic3D-1.gif)    | ![z](http://randla-net.cs.ox.ac.uk/imgs/Semantic3D-2.gif)   |
| -------------------------------- | ------------------------------- |
| ![z](imgs/Semantic3D-3.gif)    | ![z](imgs/Semantic3D-4.gif)   |



**Note:** 
- Preferably with more than 64G RAM to process this dataset due to the large volume of point cloud


### (4) SemanticKITTI

SemanticKITTI dataset can be found <a href="http://semantic-kitti.org/dataset.html#download">here</a>. Download the files
 related to semantic segmentation and extract everything into the same folder. Uncompress the folder and move it to 
`/data/semantic_kitti/dataset`.
 
- Preparing the dataset:
```
python utils/data_prepare_semantickitti.py
```

- Start training:
```
python main_SemanticKITTI.py --mode train --gpu 0
```

- Evaluation:
```
sh jobs_test_semantickitti.sh
```

Quantitative results of different approaches on SemanticKITTI dataset:

![s](http://randla-net.cs.ox.ac.uk/imgs/SemanticKITTI_table.png)

Qualitative results of our RandLA-Net:

![zzz](imgs/SemanticKITTI-2.gif)    


### (5) Demo

<p align="center"> <a href="https://youtu.be/Ar3eY_lwzMk"><img src="http://randla-net.cs.ox.ac.uk/imgs/demo_cover.png" width="80%"></a> </p>


### Citation
If you find our work useful in your research, please consider citing:

	@article{hu2019randla,
	  title={RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds},
	  author={Hu, Qingyong and Yang, Bo and Xie, Linhai and Rosa, Stefano and Guo, Yulan and Wang, Zhihua and Trigoni, Niki and Markham, Andrew},
	  journal={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
	  year={2020}
	}
	
	@article{hu2021learning,
	  title={Learning Semantic Segmentation of Large-Scale Point Clouds with Random Sampling},
	  author={Hu, Qingyong and Yang, Bo and Xie, Linhai and Rosa, Stefano and Guo, Yulan and Wang, Zhihua and Trigoni, Niki and Markham, Andrew},
	  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
	  year={2021},
	  publisher={IEEE}
	}


### Acknowledgment
-  Part of our code refers to <a href="https://github.com/jlblancoc/nanoflann">nanoflann</a> library and the the recent work <a href="https://github.com/HuguesTHOMAS/KPConv">KPConv</a>.
-  We use <a href="https://www.blender.org/">blender</a> to make the video demo.


### License
Licensed under the CC BY-NC-SA 4.0 license, see [LICENSE](./LICENSE).


### Updates
* 21/03/2020: Updating all experimental results
* 21/03/2020: Adding pretrained models and results
* 02/03/2020: Code available!
* 15/11/2019: Initial release！

## Related Repos
1. [SoTA-Point-Cloud: Deep Learning for 3D Point Clouds: A Survey](https://github.com/QingyongHu/SoTA-Point-Cloud) ![GitHub stars](https://img.shields.io/github/stars/QingyongHu/SoTA-Point-Cloud.svg?style=flat&label=Star)
2. [SensatUrban: Learning Semantics from Urban-Scale Photogrammetric Point Clouds](https://github.com/QingyongHu/SpinNet) ![GitHub stars](https://img.shields.io/github/stars/QingyongHu/SensatUrban.svg?style=flat&label=Star)
3. [3D-BoNet: Learning Object Bounding Boxes for 3D Instance Segmentation on Point Clouds](https://github.com/Yang7879/3D-BoNet) ![GitHub stars](https://img.shields.io/github/stars/Yang7879/3D-BoNet.svg?style=flat&label=Star)
4. [SpinNet: Learning a General Surface Descriptor for 3D Point Cloud Registration](https://github.com/QingyongHu/SpinNet) ![GitHub stars](https://img.shields.io/github/stars/QingyongHu/SpinNet.svg?style=flat&label=Star)
5. [SQN: Weakly-Supervised Semantic Segmentation of Large-Scale 3D Point Clouds with 1000x Fewer Labels](https://github.com/QingyongHu/SQN) ![GitHub stars](https://img.shields.io/github/stars/QingyongHu/SQN.svg?style=flat&label=Star)


