# from os.path import join, exists
# from RandLANet import Network
# from tester_Semantic3D import ModelTester
# from helper_ply import read_ply
# from helper_tool import Plot
# from helper_tool import DataProcessing as DP
# from helper_tool import mydata as cfg
# import tensorflow._api.v2.compat.v1 as tf
# tf.disable_v2_behavior()

#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# #TensorFlow按需分配显存
# config.allow_soft_placement = True
# config.log_device_placement = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.1


# import numpy as np
# import pickle, argparse, os

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.1 # maximun alloc gpu50% of MEM

# config.gpu_options.allow_growth = True #allocate dynamically


# class Semantic3D:
#     def __init__(self):
#         self.name = 'Semantic3D'
#         #self.path = '/data/semantic3d'
#         self.path = '/media/pc/C8BE4D94BE4D7BC6/data/semantic3d/'
#         self.label_to_names = {0: 'unlabeled',
#                                1: 'man-made terrain',
#                                2: 'natural terrain',
#                                3: 'high vegetation',
#                                4: 'low vegetation',
#                                5: 'buildings',
#                                6: 'hard scape',
#                                7: 'scanning artefacts',
#                                8: 'cars'}
#         self.num_classes = len(self.label_to_names)
#         self.label_values = np.sort([k for k, v in self.label_to_names.items()])
#         self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
#         self.ignored_labels = np.sort([0])
#
#         self.original_folder = join(self.path, 'original_data')    #his line need to be changed!
#         self.full_pc_folder = join(self.path, 'original_ply')      #his line need to be changed!
#         self.sub_pc_folder = join(self.path, 'input_{:.3f}'.format(cfg.sub_grid_size))
#
#         # Following KPConv to do the train-validation split
#         self.all_splits = [0, 1, 4, 5, 3, 4, 3, 0, 1, 2, 3, 4, 2, 0, 5]
#         self.val_split = 1
#
#         # Initial training-validation-testing files
#         self.train_files = []
#         self.val_files = []
#         self.test_files = []
#         cloud_names = [file_name[:-4] for file_name in os.listdir(self.original_folder) if file_name[-4:] == '.txt']
#         for pc_name in cloud_names:
#             if exists(join(self.original_folder, pc_name + '.labels')):
#                 self.train_files.append(join(self.sub_pc_folder, pc_name + '.ply'))
#             else:
#                 self.test_files.append(join(self.full_pc_folder, pc_name + '.ply'))
#
#         self.train_files = np.sort(self.train_files)
#         self.test_files = np.sort(self.test_files)
#
#         for i, file_path in enumerate(self.train_files):
#             if self.all_splits[i] == self.val_split:
#                 self.val_files.append(file_path)
#
#         self.train_files = np.sort([x for x in self.train_files if x not in self.val_files])
#
#         # Initiate containers
#         self.val_proj = []
#         self.val_labels = []
#         self.test_proj = []
#         self.test_labels = []
#
#         self.possibility = {}
#         self.min_possibility = {}
#         self.class_weight = {}
#         self.input_trees = {'training': [], 'validation': [], 'test': []}
#         self.input_colors = {'training': [], 'validation': [], 'test': []}
#         self.input_labels = {'training': [], 'validation': []}
#
#         # Ascii files dict for testing (This is make testDatasets and make these datas' labels names, these labels will generate when testing.)
#         self.ascii_files = {
#             'MarketplaceFeldkirch_Station4_rgb_intensity-reduced.ply': 'marketsquarefeldkirch4-reduced.labels',
#             'sg27_station10_rgb_intensity-reduced.ply': 'sg27_10-reduced.labels',
#             'sg28_Station2_rgb_intensity-reduced.ply': 'sg28_2-reduced.labels',
#             'StGallenCathedral_station6_rgb_intensity-reduced.ply': 'stgallencathedral6-reduced.labels',
#             'birdfountain_station1_xyz_intensity_rgb.ply': 'birdfountain1.labels',
#             'castleblatten_station1_intensity_rgb.ply': 'castleblatten1.labels',
#             'castleblatten_station5_xyz_intensity_rgb.ply': 'castleblatten5.labels',
#             'marketplacefeldkirch_station1_intensity_rgb.ply': 'marketsquarefeldkirch1.labels',
#             'marketplacefeldkirch_station4_intensity_rgb.ply': 'marketsquarefeldkirch4.labels',
#             'marketplacefeldkirch_station7_intensity_rgb.ply': 'marketsquarefeldkirch7.labels',
#             'sg27_station10_intensity_rgb.ply': 'sg27_10.labels',
#             'sg27_station3_intensity_rgb.ply': 'sg27_3.labels',
#             'sg27_station6_intensity_rgb.ply': 'sg27_6.labels',
#             'sg27_station8_intensity_rgb.ply': 'sg27_8.labels',
#             'sg28_station2_intensity_rgb.ply': 'sg28_2.labels',
#             'sg28_station5_xyz_intensity_rgb.ply': 'sg28_5.labels',
#             'stgallencathedral_station1_intensity_rgb.ply': 'stgallencathedral1.labels',
#             'stgallencathedral_station3_intensity_rgb.ply': 'stgallencathedral3.labels',
#             'stgallencathedral_station6_intensity_rgb.ply': 'stgallencathedral6.labels'}
#
#         self.load_sub_sampled_clouds(cfg.sub_grid_size)
#
#     def load_sub_sampled_clouds(self, sub_grid_size):
#
#         tree_path = join(self.path, 'input_{:.3f}'.format(sub_grid_size))
#         files = np.hstack((self.train_files, self.val_files, self.test_files))
#
#         for i, file_path in enumerate(files):
#             cloud_name = file_path.split('/')[-1][:-4]
#             print('Load_pc_' + str(i) + ': ' + cloud_name)
#             if file_path in self.val_files:
#                 cloud_split = 'validation'
#             elif file_path in self.train_files:
#                 cloud_split = 'training'
#             else:
#                 cloud_split = 'test'
#
#             # Name of the input files
#             kd_tree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
#             sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))
#
#             # read ply with data
#             data = read_ply(sub_ply_file)
#             sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
#             if cloud_split == 'test':
#                 sub_labels = None
#             else:
#                 sub_labels = data['class']
#
#             # Read pkl with search tree
#             with open(kd_tree_file, 'rb') as f:
#                 search_tree = pickle.load(f)
#
#             self.input_trees[cloud_split] += [search_tree]
#             self.input_colors[cloud_split] += [sub_colors]
#             if cloud_split in ['training', 'validation']:
#                 self.input_labels[cloud_split] += [sub_labels]
#
#         # Get validation and test re_projection indices
#         print('\nPreparing reprojection indices for validation and test')
#
#         for i, file_path in enumerate(files):
#
#             # get cloud name and split
#             cloud_name = file_path.split('/')[-1][:-4]
#
#             # Validation projection and labels
#             if file_path in self.val_files:
#                 proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
#                 with open(proj_file, 'rb') as f:
#                     proj_idx, labels = pickle.load(f)
#                 self.val_proj += [proj_idx]
#                 self.val_labels += [labels]
#
#             # Test projection
#             if file_path in self.test_files:
#                 proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
#                 with open(proj_file, 'rb') as f:
#                     proj_idx, labels = pickle.load(f)
#                 self.test_proj += [proj_idx]
#                 self.test_labels += [labels]
#         print('finished')
#         return
#
#     # Generate the input data flow
#     def get_batch_gen(self, split):
#         if split == 'training':
#             num_per_epoch = cfg.train_steps * cfg.batch_size
#         elif split == 'validation':
#             num_per_epoch = cfg.val_steps * cfg.val_batch_size
#         elif split == 'test':
#             num_per_epoch = cfg.val_steps * cfg.val_batch_size
#
#         # Reset possibility
#         self.possibility[split] = []
#         self.min_possibility[split] = []
#         self.class_weight[split] = []
#
#         # Random initialize
#         for i, tree in enumerate(self.input_trees[split]):
#             self.possibility[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
#             self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]
#
#         if split != 'test':
#             _, num_class_total = np.unique(np.hstack(self.input_labels[split]), return_counts=True)
#             self.class_weight[split] += [np.squeeze([num_class_total / np.sum(num_class_total)], axis=0)]
#
#         def spatially_regular_gen():
#
#             # Generator loop
#             for i in range(num_per_epoch):  # num_per_epoch
#
#                 # Choose the cloud with the lowest probability
#                 cloud_idx = int(np.argmin(self.min_possibility[split]))
#
#                 # choose the point with the minimum of possibility in the cloud as query point
#                 point_ind = np.argmin(self.possibility[split][cloud_idx])
#
#                 # Get all points within the cloud from tree structure
#                 points = np.array(self.input_trees[split][cloud_idx].data, copy=False)
#
#                 # Center point of input region
#                 center_point = points[point_ind, :].reshape(1, -1)
#
#                 # Add noise to the center point
#                 noise = np.random.normal(scale=cfg.noise_init / 10, size=center_point.shape)
#                 pick_point = center_point + noise.astype(center_point.dtype)
#                 query_idx = self.input_trees[split][cloud_idx].query(pick_point, k=cfg.num_points)[1][0]
#
#                 # Shuffle index
#                 query_idx = DP.shuffle_idx(query_idx)
#
#                 # Get corresponding points and colors based on the index
#                 queried_pc_xyz = points[query_idx]
#                 queried_pc_xyz[:, 0:2] = queried_pc_xyz[:, 0:2] - pick_point[:, 0:2]
#                 queried_pc_colors = self.input_colors[split][cloud_idx][query_idx]
#                 if split == 'test':
#                     queried_pc_labels = np.zeros(queried_pc_xyz.shape[0])
#                     queried_pt_weight = 1
#                 else:
#                     queried_pc_labels = self.input_labels[split][cloud_idx][query_idx]
#                     queried_pc_labels = np.array([self.label_to_idx[l] for l in queried_pc_labels])
#                     queried_pt_weight = np.array([self.class_weight[split][0][n] for n in queried_pc_labels])
#
#                 # Update the possibility of the selected points
#                 dists = np.sum(np.square((points[query_idx] - pick_point).astype(np.float32)), axis=1)
#                 delta = np.square(1 - dists / np.max(dists)) * queried_pt_weight
#                 self.possibility[split][cloud_idx][query_idx] += delta
#                 self.min_possibility[split][cloud_idx] = float(np.min(self.possibility[split][cloud_idx]))
#
#                 if True:
#                     yield (queried_pc_xyz,
#                            queried_pc_colors.astype(np.float32),
#                            queried_pc_labels,
#                            query_idx.astype(np.int32),
#                            np.array([cloud_idx], dtype=np.int32))
#
#         gen_func = spatially_regular_gen
#         gen_types = (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32)
#         gen_shapes = ([None, 3], [None, 3], [None], [None], [None])
#         return gen_func, gen_types, gen_shapes
#
#     def get_tf_mapping(self):
#         # Collect flat inputs
#         def tf_map(batch_xyz, batch_features, batch_labels, batch_pc_idx, batch_cloud_idx):
#             batch_features = tf.map_fn(self.tf_augment_input, [batch_xyz, batch_features], dtype=tf.float32)
#             input_points = []
#             input_neighbors = []
#             input_pools = []
#             input_up_samples = []
#
#             for i in range(cfg.num_layers):
#                 neigh_idx = tf.py_func(DP.knn_search, [batch_xyz, batch_xyz, cfg.k_n], tf.int32)
#                 sub_points = batch_xyz[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
#                 pool_i = neigh_idx[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
#                 up_i = tf.py_func(DP.knn_search, [sub_points, batch_xyz, 1], tf.int32)
#                 input_points.append(batch_xyz)
#                 input_neighbors.append(neigh_idx)
#                 input_pools.append(pool_i)
#                 input_up_samples.append(up_i)
#                 batch_xyz = sub_points
#
#             input_list = input_points + input_neighbors + input_pools + input_up_samples
#             input_list += [batch_features, batch_labels, batch_pc_idx, batch_cloud_idx]
#
#             return input_list
#
#         return tf_map
#
#     # data augmentation
#     @staticmethod
#     def tf_augment_input(inputs):
#         xyz = inputs[0]
#         features = inputs[1]
#         theta = tf.random_uniform((1,), minval=0, maxval=2 * np.pi)
#         # Rotation matrices
#         c, s = tf.cos(theta), tf.sin(theta)
#         cs0 = tf.zeros_like(c)
#         cs1 = tf.ones_like(c)
#         R = tf.stack([c, -s, cs0, s, c, cs0, cs0, cs0, cs1], axis=1)
#         stacked_rots = tf.reshape(R, (3, 3))
#
#         # Apply rotations
#         transformed_xyz = tf.reshape(tf.matmul(xyz, stacked_rots), [-1, 3])
#         # Choose random scales for each example
#         min_s = cfg.augment_scale_min
#         max_s = cfg.augment_scale_max
#         if cfg.augment_scale_anisotropic:
#             s = tf.random_uniform((1, 3), minval=min_s, maxval=max_s)
#         else:
#             s = tf.random_uniform((1, 1), minval=min_s, maxval=max_s)
#
#         symmetries = []
#         for i in range(3):
#             if cfg.augment_symmetries[i]:
#                 symmetries.append(tf.round(tf.random_uniform((1, 1))) * 2 - 1)
#             else:
#                 symmetries.append(tf.ones([1, 1], dtype=tf.float32))
#         s *= tf.concat(symmetries, 1)
#
#         # Create N x 3 vector of scales to multiply with stacked_points
#         stacked_scales = tf.tile(s, [tf.shape(transformed_xyz)[0], 1])
#
#         # Apply scales
#         transformed_xyz = transformed_xyz * stacked_scales
#
#         noise = tf.random_normal(tf.shape(transformed_xyz), stddev=cfg.augment_noise)
#         transformed_xyz = transformed_xyz + noise
#         rgb = features[:, :3]
#         stacked_features = tf.concat([transformed_xyz, rgb], axis=-1)
#         return stacked_features
#
#     def init_input_pipeline(self):
#         print('Initiating input pipelines')
#         cfg.ignored_label_inds = [self.label_to_idx[ign_label] for ign_label in self.ignored_labels]
#         gen_function, gen_types, gen_shapes = self.get_batch_gen('training')
#         gen_function_val, _, _ = self.get_batch_gen('validation')
#         gen_function_test, _, _ = self.get_batch_gen('test')
#         self.train_data = tf.data.Dataset.from_generator(gen_function, gen_types, gen_shapes)
#         self.val_data = tf.data.Dataset.from_generator(gen_function_val, gen_types, gen_shapes)
#         self.test_data = tf.data.Dataset.from_generator(gen_function_test, gen_types, gen_shapes)
#
#         self.batch_train_data = self.train_data.batch(cfg.batch_size)
#         self.batch_val_data = self.val_data.batch(cfg.val_batch_size)
#         self.batch_test_data = self.test_data.batch(cfg.val_batch_size)
#         map_func = self.get_tf_mapping()
#
#         self.batch_train_data = self.batch_train_data.map(map_func=map_func)
#         self.batch_val_data = self.batch_val_data.map(map_func=map_func)
#         self.batch_test_data = self.batch_test_data.map(map_func=map_func)
#
#         self.batch_train_data = self.batch_train_data.prefetch(cfg.batch_size)
#         self.batch_val_data = self.batch_val_data.prefetch(cfg.val_batch_size)
#         self.batch_test_data = self.batch_test_data.prefetch(cfg.val_batch_size)
#
#         iter = tf.data.Iterator.from_structure(self.batch_train_data.output_types, self.batch_train_data.output_shapes)
#         self.flat_inputs = iter.get_next()
#         self.train_init_op = iter.make_initializer(self.batch_train_data)
#         self.val_init_op = iter.make_initializer(self.batch_val_data)
#         self.test_init_op = iter.make_initializer(self.batch_test_data)





#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
from os.path import join, exists, dirname, abspath
from RandLANet import Network
from tester_Semantic3D import ModelTester
from helper_ply import read_ply, write_ply
from helper_tool import Plot
from helper_tool import DataProcessing as DP
from helper_tool import mydata as cfg

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import pickle, argparse, os

import open3d as o3d

from sklearn.neighbors import KDTree
import glob
import sys

from tqdm import tqdm
import datetime

BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)

def preprocess_pc(dataset_path, remove_outlier=False):
    grid_size = 0.06

    sub_pc_folder = join(dirname(dataset_path), 'input_{:.3f}'.format(grid_size))
    os.mkdir(sub_pc_folder) if not exists(sub_pc_folder) else None

    for pc_path in glob.glob(join(dataset_path, '*.txt')):
        print(pc_path)
        #file_name = pc_path.split('/')[-1][:-4]
        file_name = os.path.basename(pc_path)[:-4]

        # check if it has already calculated
        if exists(join(sub_pc_folder, file_name + '_KDTree.pkl')):
            continue

        # 使用open3d加载点云，为了后面去除离群点。This is pcd pointcloud data.
        pcd = o3d.io.read_point_cloud(pc_path, format='xyzrgb')


        if remove_outlier:
            remove_outlier_folder = join(dirname(dataset_path), 'remove_outlier')
            os.makedirs(remove_outlier_folder)

            remove_outlier_inds=join(remove_outlier_folder, file_name+".txt")
            #去除离群点
            cd,ind=pcd.remove_radius_outlier(nb_points=16, radius=10)
            inlier_cloud=pcd.select_by_index(ind)
            np.savetxt(remove_outlier_inds, ind, fmt='%d')
            pc = np.array(inlier_cloud.ponits)
        else:
            pc = np.array(pcd.points)

        #----------我自己写的代码。用于对齐pc这个数据里面的格式为xyzrgbl-----
        #mypoint_data = np.loadtxt(pc_path, delimiter=' ')
        data_list = []
        # 使用进度条读取txt文件并将数据添加到列表中
        with open(pc_path, 'r') as file:
            num_lines = sum(1 for line in file)
            file.seek(0)  # 将文件指针重新定位到文件开头
            for line in tqdm(file, total=num_lines, desc='Loading data', unit=' lines'):
                # 按空格分割行，并将数据转换为浮点数
                parts = line.strip().split()
                data_list.append([float(part) for part in parts])

        # 将数据列表转换为NumPy数组
        data = np.array(data_list)
        pc = data
        #------------------------------------------------------------


        # 我的数据集中训练和测试集都是x,y,z,r,g,b,label的存储格式
        labels = pc[:, -1].astype(np.uint8)

        # save sub_cloud and KDTree file
        sub_xyz, sub_colors, sub_labels = DP.grid_sub_sampling(pc[:, :3].astype(np.float32), pc[:, 3:6].astype(np.uint8), labels, grid_size)
        # sub_xyz, sub_colors, sub_labels = DP.grid_sub_sampling(pc[:, :3].astype(np.float32), pc[:, 4:7].astype(np.uint8), labels, grid_size)
        sub_colors = sub_colors / 255.0
        sub_labels = np.squeeze(sub_labels)
        sub_ply_file = join(sub_pc_folder, file_name + '.ply')
        write_ply(sub_ply_file, [sub_xyz, sub_colors, sub_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])
        #print(sub_ply_file)

        search_tree = KDTree(sub_xyz, leaf_size=50)
        kd_tree_file = join(sub_pc_folder, file_name + '_KDTree.pkl')
        with open(kd_tree_file, 'wb') as f:
            pickle.dump(search_tree, f)

        proj_idx = np.squeeze(search_tree.query(pcd.points, return_distance=False))
        proj_idx = proj_idx.astype(np.int32)
        proj_save = join(sub_pc_folder, file_name + '_proj.pkl')
        with open(proj_save, 'wb') as f:
            pickle.dump([proj_idx, labels], f)


class mydata:
    def __init__(self, path, remove_outlier=False):
        self.name = 'mydata'
        self.path = '/media/pc/C8BE4D94BE4D7BC6/data/LYYData4/LYYinput/'
        # self.path = path
        self.label_to_names = {0: 'SteelStructureFrame',
                               1: 'Land',
                               2: 'TowerCrane',
                               3: 'SupportFrame',
                               4: 'ConstructionVehicle',
                               5: 'Dome'
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

        #-------------
        current_datetime = datetime.datetime.now()
        self.val_split = [current_datetime]

        TestPath1 = '/media/pc/C8BE4D94BE4D7BC6/data/LYYData4/LYYinput/input_0.060/testData1.ply'
        TestPath2 = '/media/pc/C8BE4D94BE4D7BC6/data/LYYData4/LYYinput/input_0.060/testData2.ply'
        TestPath3 = '/media/pc/C8BE4D94BE4D7BC6/data/LYYData4/LYYinput/input_0.060/testData3.ply'
        self.test_file = [TestPath1, TestPath2, TestPath3]
        #-------------

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
        # Collect flat inputs
        def tf_map(batch_xyz, batch_features, batch_labels, batch_pc_idx, batch_cloud_idx):
            batch_features = tf.map_fn(self.tf_augment_input, [batch_xyz, batch_features], dtype=tf.float32)
            input_points = []
            input_neighbors = []
            input_pools = []
            input_up_samples = []

            for i in range(cfg.num_layers):
                neigh_idx = tf.py_func(DP.knn_search, [batch_xyz, batch_xyz, cfg.k_n], tf.int32)
                sub_points = batch_xyz[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                pool_i = neigh_idx[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                up_i = tf.py_func(DP.knn_search, [sub_points, batch_xyz, 1], tf.int32)
                input_points.append(batch_xyz)
                input_neighbors.append(neigh_idx)
                input_pools.append(pool_i)
                input_up_samples.append(up_i)
                batch_xyz = sub_points

            input_list = input_points + input_neighbors + input_pools + input_up_samples
            input_list += [batch_features, batch_labels, batch_pc_idx, batch_cloud_idx]

            return input_list

        return tf_map

    # data augmentation
    @staticmethod
    def tf_augment_input(inputs):
        xyz = inputs[0]
        features = inputs[1]
        theta = tf.random_uniform((1,), minval=0, maxval=2 * np.pi)
        # Rotation matrices
        c, s = tf.cos(theta), tf.sin(theta)
        cs0 = tf.zeros_like(c)
        cs1 = tf.ones_like(c)
        R = tf.stack([c, -s, cs0, s, c, cs0, cs0, cs0, cs1], axis=1)
        stacked_rots = tf.reshape(R, (3, 3))

        # Apply rotations
        transformed_xyz = tf.reshape(tf.matmul(xyz, stacked_rots), [-1, 3])
        # Choose random scales for each example
        min_s = cfg.augment_scale_min
        max_s = cfg.augment_scale_max
        if cfg.augment_scale_anisotropic:
            s = tf.random_uniform((1, 3), minval=min_s, maxval=max_s)
        else:
            s = tf.random_uniform((1, 1), minval=min_s, maxval=max_s)

        symmetries = []
        for i in range(3):
            if cfg.augment_symmetries[i]:
                symmetries.append(tf.round(tf.random_uniform((1, 1))) * 2 - 1)
            else:
                symmetries.append(tf.ones([1, 1], dtype=tf.float32))
        s *= tf.concat(symmetries, 1)

        # Create N x 3 vector of scales to multiply with stacked_points
        stacked_scales = tf.tile(s, [tf.shape(transformed_xyz)[0], 1])

        # Apply scales
        transformed_xyz = transformed_xyz * stacked_scales

        noise = tf.random_normal(tf.shape(transformed_xyz), stddev=cfg.augment_noise)
        transformed_xyz = transformed_xyz + noise
        rgb = features[:, :3]
        stacked_features = tf.concat([transformed_xyz, rgb], axis=-1)
        return stacked_features


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

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------



# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--gpu', type=int, default=0, help='the number of GPUs to use [default: 0]')
#     parser.add_argument('--mode', type=str, default='train', help='options: train, test, vis')
#     parser.add_argument('--model_path', type=str, default='None', help='pretrained model path')
#     FLAGS = parser.parse_args()
#
#     GPU_ID = FLAGS.gpu
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)
#     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#
#     Mode = FLAGS.mode
#     dataset = Semantic3D()
#     dataset.init_input_pipeline()
#
#     if Mode == 'train':
#         model = Network(dataset, cfg)
#         model.train(dataset)
#     elif Mode == 'test':
#         cfg.saving = False
#         model = Network(dataset, cfg)
#         if FLAGS.model_path is not 'None':
#             chosen_snap = FLAGS.model_path
#         else:
#             chosen_snapshot = -1
#             logs = np.sort([os.path.join('results', f) for f in os.listdir('results') if f.startswith('Log')])
#             chosen_folder = logs[-1]
#             snap_path = join(chosen_folder, 'snapshots')
#             snap_steps = [int(f[:-5].split('-')[-1]) for f in os.listdir(snap_path) if f[-5:] == '.meta']
#             chosen_step = np.sort(snap_steps)[-1]
#             chosen_snap = os.path.join(snap_path, 'snap-{:d}'.format(chosen_step))
#         tester = ModelTester(model, dataset, restore_snap=chosen_snap)
#         tester.test(model, dataset)
#
#     else:
#         ##################
#         # Visualize data #
#         ##################
#
#         with tf.Session() as sess:
#             sess.run(tf.global_variables_initializer())
#             sess.run(dataset.train_init_op)
#             while True:
#                 flat_inputs = sess.run(dataset.flat_inputs)
#                 pc_xyz = flat_inputs[0]
#                 sub_pc_xyz = flat_inputs[1]
#                 labels = flat_inputs[21]
#                 Plot.draw_pc_sem_ins(pc_xyz[0, :, :], labels[0, :])
#                 Plot.draw_pc_sem_ins(sub_pc_xyz[0, :, :], labels[0, 0:np.shape(sub_pc_xyz)[1]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/media/pc/C8BE4D94BE4D7BC6/data/LYYData4/LYYinput/')   #Default:None   /media/pc/C8BE4D94BE4D7BC6/data/LYYData4/LYYinput/
    parser.add_argument('--mode', type=str, default='test', help='options: train, test, vis')  #only can test!!!
    parser.add_argument('--output_dir', type=str, default='/media/pc/C8BE4D94BE4D7BC6/data/LYYData4/LYYoutput/') #Default:None   /media/pc/C8BE4D94BE4D7BC6/data/LYYData4/LYYoutput/
    parser.add_argument('--gpu', type=int, default=0, help='the number of GPUs to use [default: 0]')
    parser.add_argument('--remove_outlier', type=bool, default=False, help='options: True, False')
    parser.add_argument('--model_path', type=str, default='/home/pc/myGitHub/RandLA-Net/results/Log_2023-10-01_12-40-52/snapshots/snap-11001', help='pretrained model path')              #Default:None   /home/pc/myGitHub/RandLA-Net/results/Log_2023-10-01_12-40-52/snapshots/snap-11001
    FLAGS = parser.parse_args()

    GPU_ID = FLAGS.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    origin_data = os.path.join(FLAGS.input_dir, "original_data")

    preprocess_pc(origin_data, FLAGS.remove_outlier)

    Mode = FLAGS.mode
    dataset = mydata(FLAGS.input_dir, FLAGS.remove_outlier)
    dataset.init_input_pipeline()

    cfg.saving = False
    # 网络初始化
    model = Network(dataset, cfg)
    tester = ModelTester(model, dataset, restore_snap=FLAGS.model_path)
    # 开始测试
    tester.test(model, dataset, FLAGS.output_dir)

