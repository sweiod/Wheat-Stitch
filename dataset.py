from torch.utils.data import Dataset
import numpy as np
import cv2, torch
import os
import glob
from collections import OrderedDict
import random


class TrainDataset(Dataset):
    def __init__(self, data_path):
        self.width = 512
        self.height = 512
        self.train_path = data_path
        self.datas = OrderedDict()

        datas = glob.glob(os.path.join(self.train_path, '*'))
        print(f"Datasets found: {datas}")
        for data in sorted(datas):
            data_name = os.path.basename(data)
            if data_name == 'input1' or data_name == 'input2':
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                self.datas[data_name]['image'] = glob.glob(os.path.join(data, '*.jpg'))
                self.datas[data_name]['image'].sort()
                print(f"Loaded {len(self.datas[data_name]['image'])} images from {data}")
        print("Data keys:", self.datas.keys())

    def __getitem__(self, index):
        input1 = cv2.imread(self.datas['input1']['image'][index])
        input1 = cv2.resize(input1, (self.width, self.height))
        input1 = input1.astype(dtype=np.float32)
        input1 = (input1 / 127.5) - 1.0
        input1 = np.transpose(input1, [2, 0, 1])

        input2 = cv2.imread(self.datas['input2']['image'][index])
        input2 = cv2.resize(input2, (self.width, self.height))
        input2 = input2.astype(dtype=np.float32)
        input2 = (input2 / 127.5) - 1.0
        input2 = np.transpose(input2, [2, 0, 1])

        input1_tensor = torch.tensor(input1)
        input2_tensor = torch.tensor(input2)

        if_exchange = random.randint(0, 1)
        if if_exchange == 0:
            return (input1_tensor, input2_tensor)
        else:
            return (input2_tensor, input1_tensor)

    def __len__(self):
        return len(self.datas['input1']['image'])


class TestDataset(Dataset):
    def __init__(self, data_path):
        self.width = 512
        self.height = 512
        self.test_path = data_path
        self.datas = OrderedDict()

        datas = glob.glob(os.path.join(self.test_path, '*'))
        print(f"Datasets found: {datas}")
        for data in sorted(datas):
            data_name = os.path.basename(data)
            if data_name == 'input1' or data_name == 'input2':
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                self.datas[data_name]['image'] = glob.glob(os.path.join(data, '*.jpg'))
                self.datas[data_name]['image'].sort()
                print(f"Loaded {len(self.datas[data_name]['image'])} images from {data}")
        print("Data keys:", self.datas.keys())

    def __getitem__(self, index):
        input1 = cv2.imread(self.datas['input1']['image'][index])
        input1 = input1.astype(dtype=np.float32)
        input1 = (input1 / 127.5) - 1.0
        input1 = np.transpose(input1, [2, 0, 1])

        input2 = cv2.imread(self.datas['input2']['image'][index])
        input2 = input2.astype(dtype=np.float32)
        input2 = (input2 / 127.5) - 1.0
        input2 = np.transpose(input2, [2, 0, 1])

        input1_tensor = torch.tensor(input1)
        input2_tensor = torch.tensor(input2)

        return (input1_tensor, input2_tensor)

    def __len__(self):
        return len(self.datas['input1']['image'])




class TrainDataset(Dataset):
    def __init__(self, data_path):
        self.width = 512
        self.height = 512
        self.train_path = data_path
        self.datas = OrderedDict()

        datas = glob.glob(os.path.join(self.train_path, '*'))
        print(f"Datasets found: {datas}")
        for data in sorted(datas):
            data_name = os.path.basename(data)
            if data_name == 'input1' or data_name == 'input2':
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                self.datas[data_name]['image'] = glob.glob(os.path.join(data, '*.jpg'))
                self.datas[data_name]['image'].sort()
                print(f"Loaded {len(self.datas[data_name]['image'])} images from {data}")
        print("Data keys:", self.datas.keys())

    def __getitem__(self, index):
        input1 = cv2.imread(self.datas['input1']['image'][index])
        input1 = cv2.resize(input1, (self.width, self.height))
        input1 = input1.astype(dtype=np.float32)
        input1 = (input1 / 127.5) - 1.0
        input1 = np.transpose(input1, [2, 0, 1])

        input2 = cv2.imread(self.datas['input2']['image'][index])
        input2 = cv2.resize(input2, (self.width, self.height))
        input2 = input2.astype(dtype=np.float32)
        input2 = (input2 / 127.5) - 1.0
        input2 = np.transpose(input2, [2, 0, 1])

        input1_tensor = torch.tensor(input1)
        input2_tensor = torch.tensor(input2)

        if_exchange = random.randint(0, 1)
        if if_exchange == 0:
            return (input1_tensor, input2_tensor)
        else:
            return (input2_tensor, input1_tensor)

    def __len__(self):
        return len(self.datas['input1']['image'])


class TestDataset(Dataset):
    def __init__(self, data_path):
        self.width = 512
        self.height = 512
        self.test_path = data_path
        self.datas = OrderedDict()

        datas = glob.glob(os.path.join(self.test_path, '*'))
        print(f"Datasets found: {datas}")
        for data in sorted(datas):
            data_name = os.path.basename(data)
            if data_name == 'input1' or data_name == 'input2':
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                self.datas[data_name]['image'] = glob.glob(os.path.join(data, '*.jpg'))
                self.datas[data_name]['image'].sort()
                print(f"Loaded {len(self.datas[data_name]['image'])} images from {data}")
        print("Data keys:", self.datas.keys())

    def __getitem__(self, index):
        input1 = cv2.imread(self.datas['input1']['image'][index])
        input1 = input1.astype(dtype=np.float32)
        input1 = (input1 / 127.5) - 1.0
        input1 = np.transpose(input1, [2, 0, 1])

        input2 = cv2.imread(self.datas['input2']['image'][index])
        input2 = input2.astype(dtype=np.float32)
        input2 = (input2 / 127.5) - 1.0
        input2 = np.transpose(input2, [2, 0, 1])

        input1_tensor = torch.tensor(input1)
        input2_tensor = torch.tensor(input2)

        return (input1_tensor, input2_tensor)

    def __len__(self):
        return len(self.datas['input1']['image'])
