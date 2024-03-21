#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
    @Project ：f 
    @File    ：dataset.py
    @IDE     ：PyCharm 
    @Author  ：kemove
    @Date    ：2023/9/22 下午8:05 
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class CustomDataset1(Dataset):
    def __init__(self, root_dir1, root_dir2, root_dir3):
        self.data = []  # 存储数据集
        self.data1 = []  # 存储数据集
        self.label = []  # 存储数据集
        self.norm_param = {
                    'dBZ': [0, 65],
                    'ZDR': [-1, 5],
                    'KDP': [-1, 6],
                    'NJU_CPOL_kdpRain':[0, 200],
        }

        # 获取一级目录下的文件夹列表
        subdirectories1 = sorted([d for d in os.listdir(root_dir1) if os.path.isdir(os.path.join(root_dir1, d))])
        subdirectories2 = sorted([d for d in os.listdir(root_dir2) if os.path.isdir(os.path.join(root_dir2, d))])
        subdirectories3 = sorted([d for d in os.listdir(root_dir3) if os.path.isdir(os.path.join(root_dir3, d))])

        # 遍历三个根目录下的子目录
        for subdir1, subdir2, subdir3 in zip(subdirectories1, subdirectories2, subdirectories3):
            subdir_path1 = os.path.join(root_dir1, subdir1)
            subdir_path2 = os.path.join(root_dir2, subdir2)
            subdir_path3 = os.path.join(root_dir3, subdir3)

            # 获取子文件夹中所有.npy文件
            npy_files1 = sorted([f for f in os.listdir(subdir_path1) if f.endswith('.npy')])[:20]
            npy_files2 = sorted([f for f in os.listdir(subdir_path2) if f.endswith('.npy')])[:20]
            npy_files3 = sorted([f for f in os.listdir(subdir_path3) if f.endswith('.npy')])[:20]

            # 逐个加载.npy文件并添加到列表


            # 归一化处理
            feature1 = root_dir1.split('/')[-2]
            mmin1, mmax1 = self.norm_param[feature1]

            feature2 = root_dir2.split('/')[-2]
            mmin2, mmax2 = self.norm_param[feature2]

            feature3 = root_dir3.split('/')[-1]
            mmin3, mmax3 = self.norm_param[feature3]

            for f in npy_files1:
                data = np.load(os.path.join(subdir_path1, f))
                x0 = (data - mmin1) / (mmax1 - mmin1)
                self.data.append(x0)

            for f in npy_files2:
                data = np.load(os.path.join(subdir_path2, f))
                x1 = (data - mmin2) / (mmax2 - mmin2)
                self.data1.append(x1)

            for f in npy_files3:
                data = np.load(os.path.join(subdir_path3, f))
                y = (data - mmin3) / (mmax1 - mmin3)
                self.label.append(y)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取数据和标签
        x0 = self.data[idx]
        x1 = self.data1[idx]
        y = self.label[idx]

        # 将数据转换为 NumPy 数组
        x00 = np.array(x0)
        x11 = np.array(x1)

        # 合并数据通道
        x = np.concatenate([x00[np.newaxis, :], x11[np.newaxis, :]], axis=0)

        # 转换为 PyTorch 张量
        x = torch.tensor(np.array(x), dtype=torch.float32)
        y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(0)  # 添加批次维度
        return x, y

class CustomDataset2(Dataset):
    def __init__(self, root_dir1, root_dir2, root_dir3, num_samples=10):
        self.data1 = []  # 存储数据集
        self.data2 = []  # 存储数据集
        self.data3 = []  # 存储数据集
        self.label = []  # 存储数据集
        self.num_samples = num_samples
        self.norm_param = {
                    'dBZ': [0, 65],
                    'ZDR': [-1, 5],
                    'KDP': [-1, 6],
                    'NJU_CPOL_kdpRain':[0, 200],
        }

        # 获取一级目录下的文件夹列表
        subdirectories1 = sorted([d for d in os.listdir(root_dir1) if os.path.isdir(os.path.join(root_dir1, d))])
        subdirectories2 = sorted([d for d in os.listdir(root_dir2) if os.path.isdir(os.path.join(root_dir2, d))])
        subdirectories3 = sorted([d for d in os.listdir(root_dir3) if os.path.isdir(os.path.join(root_dir3, d))])

        # 遍历三个根目录下的子目录
        for subdir1, subdir2, subdir3 in zip(subdirectories1, subdirectories2, subdirectories3):
            subdir_path1 = os.path.join(root_dir1, subdir1)
            subdir_path2 = os.path.join(root_dir2, subdir2)
            subdir_path3 = os.path.join(root_dir3, subdir3)

            # 获取子文件夹中所有.npy文件
            npy_files1 = sorted([f for f in os.listdir(subdir_path1) if f.endswith('.npy')])[:20]
            npy_files2 = sorted([f for f in os.listdir(subdir_path2) if f.endswith('.npy')])[:20]
            npy_files3 = sorted([f for f in os.listdir(subdir_path3) if f.endswith('.npy')])[:20]

            # 归一化处理
            feature1 = root_dir1.split('/')[-2]
            mmin1, mmax1 = self.norm_param[feature1]

            feature2 = root_dir2.split('/')[-2]
            mmin2, mmax2 = self.norm_param[feature2]

            feature3 = root_dir3.split('/')[-2]
            mmin3, mmax3 = self.norm_param[feature3]


            x1 = []
            x2 = []
            x3 = []
            y = []

            # 遍历.npy文件，选择前10个和后10个
            for i, npy_file in enumerate(npy_files1):
                data = np.load(os.path.join(subdir_path1, npy_file))
                x = (data - mmin1) / (mmax1 - mmin1)
                if i < num_samples:
                    x1.append(x)
                elif i < 2 * num_samples:
                    y.append(x)
            self.data1.append(x1)
            self.label.append(y)

            for i, npy_file in enumerate(npy_files2):
                data = np.load(os.path.join(subdir_path2, npy_file))
                x = (data - mmin2) / (mmax2 - mmin2)
                if i < num_samples:
                    x2.append(x)
            self.data2.append(x2)


            for i, npy_file in enumerate(npy_files3):
                data = np.load(os.path.join(subdir_path3, npy_file))
                x = (data - mmin3) / (mmax3 - mmin3)
                if i < num_samples:
                    x3.append(x)
            self.data3.append(x3)

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, idx):
        # 获取数据和标签
        x1 = self.data1[idx]
        x2 = self.data2[idx]
        x3 = self.data3[idx]
        y = self.label[idx]

        # 将数据转换为 NumPy 数组
        x11 = np.array(x1)
        x22 = np.array(x2)
        x33 = np.array(x3)

        # 合并数据通道
        # x = np.concatenate([x11[np.newaxis, :], x22[np.newaxis, :], x33[np.newaxis, :]], axis=0)
        x = np.concatenate([x11, x22, x33], axis=0)

        # 转换为 PyTorch 张量
        x = torch.tensor(np.array(x), dtype=torch.float32)
        y = torch.tensor(np.array(y), dtype=torch.float32)  # 添加批次维度
        return x, y




def main():
    # 用法示例
    root_directory = '/home/kemove/data/datasets/f题/NJU_CPOL_update2308/dBZ/1.0km'
    root_directory1 = '/home/kemove/data/datasets/f题/NJU_CPOL_update2308/ZDR/1.0km'
    root_directory3 = '/home/kemove/data/datasets/f题/NJU_CPOL_update2308/KDP/1.0km'
    root_directory2 = '/home/kemove/data/datasets/f题/NJU_CPOL_kdpRain'
    # ds = CustomDataset1(root_directory, root_directory1, root_directory2)
    ds = CustomDataset2(root_directory, root_directory1, root_directory3)
    print(ds)
    dl = DataLoader(ds, shuffle=False, batch_size=1, num_workers=8, drop_last=False)
    for step, (inputs, labels) in enumerate(dl):
        inputs = inputs
        labels = labels
        print("------------")



if __name__ == '__main__':
    main()
