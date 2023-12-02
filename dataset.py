# Developed by Liguang Zhou, 2020.9.17

import json
import torch
import torchvision.datasets as datasets
import torchvision.models as models
import os
import numpy as np
from model.data_partition import esc_dataset_construction, dcase_dataset_construction, home_automation_dataset_construction, data_construction

class DatasetSelection(object):
    def datasetSelection(self, args):
        audio_datasets = ['ESC10', 'ESC50', 'DCASE2019', 'US8K', 'HomeAutomation', 'DCASE2021', 'DCASE2021-Visual', 'DCASE2021-Audio-Visual', 'ADVANCE']

        if args.dataset_name in audio_datasets:
            if args.dataset_name == 'US8K':
                csv_file = '/data/lgzhou/dataset/UrbanSound8K/metadata/UrbanSound8K.csv'
                data_dir = '/data/lgzhou/dataset/UrbanSound8K/audio/'
                data_sample = esc_dataset_construction(csv_file, args.test_set_id, args.dataset_name)

            elif args.dataset_name == 'ESC10' or args.dataset_name =='ESC50':
                csv_file = '/data/lgzhou/dataset/ESC-50/meta/esc50.csv'
                data_dir = '/data/lgzhou/dataset/ESC-50/audio/'
                data_sample = esc_dataset_construction(csv_file, args.test_set_id, args.dataset_name)

            elif args.dataset_name == 'DCASE2019':
                train_csv_file = '/data/lgzhou/dataset/TAU-urban-acoustic-scenes-2019-development/evaluation_setup/fold1_train.csv'
                val_csv_file = '/data/lgzhou/dataset/TAU-urban-acoustic-scenes-2019-development/evaluation_setup/fold1_evaluate.csv'
                data_dir = '/data/lgzhou/dataset/TAU-urban-acoustic-scenes-2019-development/'
                data_sample = dcase_dataset_construction(args, train_csv_file, val_csv_file)

            elif args.dataset_name == 'DCASE2021' or args.dataset_name == 'DCASE2021-Visual' or args.dataset_name == 'DCASE2021-Audio-Visual':
                train_csv_file = '/data/lgzhou/dataset/DCASE2021/evaluation_setup/fold1_train.csv'
                val_csv_file = '/data/lgzhou/dataset/DCASE2021/evaluation_setup/fold1_evaluate.csv'
                data_dir = '/data/lgzhou/dataset/DCASE2021/'
                data_sample = dcase_dataset_construction(args, train_csv_file, val_csv_file)

            elif args.dataset_name == 'ADVANCE':
                data_dir = '/data/lgzhou/dataset/ADVANCE/vision'
                data_sample = data_construction(data_dir)


            elif args.dataset_name == 'HomeAutomation':
                train_csv_file = '/data/lgzhou/dataset/Home_automation/meta/homeautomation.csv'
                data_dir = '/data/lgzhou/dataset/Home_automation/audio/'
                data_sample = home_automation_dataset_construction(train_csv_file)

            return data_dir, data_sample

        else:
            if(args.dataset_name == 'Places365-7'):
                # Data directory
                data_dir = '/data/dataset/Places365-7'

                # load the dictionary which contains objects for every image in dataset
                # one_hot = self.load_dict('object_information/150obj_Places365_7.json')

                # class information
                # file_name = './object_information/categories_Places365_7.txt'
                # classes = list()
                # with open(file_name) as class_file:
                #     for line in class_file:
                #         classes.append(line.strip().split(' ')[0][3:])
                # classes = tuple(classes)

            elif(args.dataset_name == 'Places365-14'):
                # Data directory
                data_dir = '/data/dataset/Places365-14'

                # load the dictionary which contains objects for every image in dataset
                # one_hot = self.load_dict('object_information/150obj_Places365_14.json')

                # class information
                # file_name = './object_information/categories_Places365_14.txt'
                # classes = list()
                # with open(file_name) as class_file:
                #     for line in class_file:
                #         classes.append(line.strip().split(' ')[0][3:])
                # classes = tuple(classes)

            elif(args.dataset_name == 'SUNRGBD'):
                # Data directory
                data_dir = '/data/dataset/SUNRGBD'

                # load the dictionary which contains objects for every image in dataset
                # one_hot = self.load_dict('object_information/150obj_7classes_SUN.json')

                # class information
                # file_name = './object_information/categories_Places365_7.txt'
                # classes = list()
                # with open(file_name) as class_file:
                #     for line in class_file:
                #         classes.append(line.strip().split(' ')[0][3:])
                # classes = tuple(classes)
            elif(args.dataset_name == 'NYUdata'):
                data_dir = '/data/dataset/NYUdata'


            elif (args.dataset_name == 'MIT67'):
                # Data directory
                data_dir = '/data/dataset/MIT67'
                # class information
                # file_name = './object_information/categories_Places365_7.txt'
                # classes = list()
                # with open(file_name) as class_file:
                #     for line in class_file:
                #         classes.append(line.strip().split(' ')[0][3:])
                # classes = tuple(classes)
            elif (args.dataset_name == 'SUN_RGBD'):
                data_dir = '/data/dataset/SUN_RGBD'

            elif (args.dataset_name == 'NYUdata'):
                data_dir = '/data/dataset/NYUdata'
            
        return data_dir

    def load_dict(self, filename):
        with open(filename, "r") as json_file:
            dic = json.load(json_file)
        return dic

    def discriminative_matrix_estimation(self):
        # create p_o_c matrix
        if (self.dataset_name == 'Places365-7' or self.dataset_name == 'SUNRGBD'):
            fileName = './object_information/150obj_result_Places365_7.npy'
            self.num_sp = np.load(fileName)
            fileName = './object_information/150obj_number_Places365_7.npy'
            self.num_total = np.load(fileName)
            self.cls_num = 7
            self.obj_num = 150

        elif (self.dataset_name == 'Places365-14'):
            fileName = './object_information/150obj_result_Places365_14.npy'
            self.num_sp = np.load(fileName)
            fileName = './object_information/150obj_number_Places365_14.npy'
            self.num_total = np.load(fileName)
            self.cls_num = 14
            self.obj_num = 150

        matrix_p_o_c = np.zeros(shape=(self.cls_num, self.obj_num, self.obj_num))

        for i in range(self.cls_num):
            X = []
            Y = []
            Z = []
            p_o_c = self.num_sp[i] / self.num_total[i]
            p_o_c = p_o_c.reshape(1, p_o_c.shape[0])
            #    print(p_o_c)
            p_o_c_tran = p_o_c.T
            #    print(p_o_c_tran)
            matrix_p_o_c[i] = np.dot(p_o_c_tran, p_o_c)

        matrix_p_c_o = np.zeros(shape=(self.cls_num, self.obj_num, self.obj_num))
        discriminative_matrix = np.zeros(shape=(self.obj_num, self.obj_num))
        temp = np.zeros(shape=self.cls_num)
        for i in range(self.obj_num):
            for j in range(self.obj_num):
                sum = 0
                for k in range(self.cls_num):
                    sum += matrix_p_o_c[k][i][j] * 1 / self.cls_num
                if sum == 0:
                    matrix_p_c_o[k][i][j] = 0
                    continue
                for k in range(self.cls_num):
                    matrix_p_c_o[k][i][j] = matrix_p_o_c[k][i][j] * 1 / self.cls_num / sum
                    temp[k] = matrix_p_c_o[k][i][j]
                discriminative_matrix[i][j] = temp.std()

        # print('discriminative_matrix:', discriminative_matrix.shape, discriminative_matrix)
        return discriminative_matrix

class ImageFolderWithPaths(datasets.ImageFolder):
        """Custom dataset that includes image file paths. Extends
        torchvision.datasets.ImageFolder
        """

        # override the __getitem__ method. this is the method dataloader calls
        def __getitem__(self, index):
            # this is what ImageFolder normally returns 
            original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
            # the image file path
            path = self.imgs[index][0]
            # make a new tuple that includes original and the path
            tuple_with_path = (original_tuple + (path,))
            return tuple_with_path