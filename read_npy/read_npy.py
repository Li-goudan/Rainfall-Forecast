#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
    @Project ：f 
    @File    ：read_npy.py
    @IDE     ：PyCharm 
    @Author  ：kemove
    @Date    ：2023/9/25 下午7:38 
"""
import os
import numpy as np
import matplotlib.pyplot as plt

subdir_path1 = "./data_dir_000_1"
subdir_path2 = "./data_dir_000_3"
subdir_path3 = "./data_dir_000_7"
subdir_path4 = "./data_dir_000_rain"

def read_npy(subdir_path1, subdir_path2, subdir_path3, subdir_path4):
	# 获取子文件夹中所有.npy文件
	npy_files1 = sorted([f for f in os.listdir(subdir_path1) if f.endswith('.npy')])[:20]
	npy_files2 = sorted([f for f in os.listdir(subdir_path2) if f.endswith('.npy')])[:20]
	npy_files3 = sorted([f for f in os.listdir(subdir_path3) if f.endswith('.npy')])[:20]
	npy_files4 = sorted([f for f in os.listdir(subdir_path3) if f.endswith('.npy')])[:20]

	for i, (npy_file1, npy_file2, npy_file3, npy_file4) in enumerate(zip(npy_files3, npy_files2, npy_files3, npy_files4)):
		data1 = np.load(os.path.join(subdir_path1, npy_file1))
		data2 = np.load(os.path.join(subdir_path2, npy_file2))
		data3 = np.load(os.path.join(subdir_path3, npy_file3))
		data4 = np.load(os.path.join(subdir_path4, npy_file4))

		# 归一化
		data1 = (data1 - np.min(data1)) / (np.max(data1) - np.min(data1))
		data2 = (data2 - np.min(data2)) / (np.max(data2) - np.min(data2))
		data3 = (data3 - np.min(data3)) / (np.max(data3) - np.min(data3))
		data4 = (data4 - np.min(data4)) / (np.max(data4) - np.min(data4))

		plt.figure(figsize=(10, 9))
		# plt.figure(figsize=(15, 4))

		# 绘制第一个图片）
		plt.subplot(2, 2, 1)
		plt.title('$Z_{H}$')
		plt.imshow(data1, cmap='viridis', extent=[-128, 127, -128, 127])
		# plt.imshow(img0, cmap='gray')
		# # 添加颜色条
		plt.colorbar()

		# 添加横纵坐标轴标签
		# plt.xlabel('X (Km)')
		plt.ylabel('Y (Km)')

		# 绘制第二个图片
		plt.subplot(2, 2, 2)
		plt.title('$Z_{DR}$')
		plt.imshow(data2, cmap='viridis', extent=[-128, 127, -128, 127])
		# 添加颜色条
		plt.colorbar()
		# plt.xlabel('X (Km)')

		# 绘制第3个图片
		plt.subplot(2, 2, 3)
		plt.title('$K_{DP}$')
		plt.imshow(data3, cmap='viridis', extent=[-128, 127, -128, 127])
		# plt.imshow(img2, cmap='gray')
		# 添加颜色条
		plt.colorbar()
		plt.xlabel('X (Km)')
		plt.ylabel('Y (Km)')

		# 绘制第4个图片
		plt.subplot(2, 2, 4)
		plt.title('Real Rain')
		plt.imshow(data4, cmap='viridis', extent=[-128, 127, -128, 127])
		# plt.imshow(img2, cmap='gray')
		# 添加颜色条
		plt.colorbar()
		plt.xlabel('X (Km)')

		# # 绘制第一个图片（左侧子图）
		# plt.subplot(1, 3, 1)  # 1行2列，第1个子图
		# plt.title('1 Km')
		# plt.imshow(data1, cmap='viridis', extent=[-128, 127, -128, 127])
		# # plt.imshow(img0, cmap='gray')
		# # # 添加颜色条
		#
		# plt.colorbar()
		# # plt.colorbar(location='right')
		#
		# # 添加横纵坐标轴标签
		# plt.xlabel('X (Km)')
		# plt.ylabel('Y (Km)')
		#
		# # 绘制第二个图片（中间子图）
		# plt.subplot(1, 3, 2)  # 1行2列，第2个子图
		# plt.title('3 Km')
		# plt.imshow(data2, cmap='viridis', extent=[-128, 127, -128, 127])
		# # 添加颜色条
		# plt.colorbar()
		# plt.xlabel('X (Km)')
		#
		# # 绘制第3个图片（右侧子图）
		# plt.subplot(1, 3, 3)  # 1行3列，第3个子图
		# plt.title('7 Km')
		# plt.imshow(data3, cmap='viridis', extent=[-128, 127, -128, 127])
		# # plt.imshow(img2, cmap='gray')
		# # 添加颜色条
		# plt.colorbar()
		# plt.xlabel('X (Km)')


		plt.savefig(os.path.join("output_norm", f"reconstructed_{i}.png"), dpi=300)

		plt.close()
