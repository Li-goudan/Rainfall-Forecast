#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
    @Project ：f 
    @File    ：test.py
    @IDE     ：PyCharm 
    @Author  ：kemove
    @Date    ：2023/9/22 下午9:28 
"""
import os
import torch
from torch.utils.data import DataLoader
from unet import UNet
# from UNet import UNet
from data.dataset import CustomDataset
from data.dataset import CustomDataset1
from data.dataset import CustomDataset2
import numpy as np
from torch import nn
from loss_ssim import SSIMLoss

import matplotlib.pyplot as plt


def savefig(inputs, labels, predicts, i, output_path):
    # 可视化输入以及预测输出
    input = inputs.view(2, 256, 256).cpu().numpy()
    label = labels.view(1, 256, 256).cpu().numpy()
    pred = predicts.view(1, 256, 256).cpu().numpy()

    img0 = input[0, :, :]
    img3 = input[1, :, :]
    img1 = label[0, :, :]
    img2 = pred[0, :, :]


    # # 绘制第一个图片（左侧子图）
    # plt.subplot(1, 3, 1)  # 1行2列，第1个子图
    # plt.title('Input(1st Frame)')
    # plt.imshow(img0, cmap='viridis', extent=[-128,127,-128,127])
    # # plt.imshow(img0, cmap='gray') 
    # # # 添加颜色条
    # plt.colorbar()
    #
    # # 添加横纵坐标轴标签
    # plt.xlabel('X (Km)')
    # plt.ylabel('Y (Km)')
    #
    # # 绘制第二个图片（中间子图）
    # plt.subplot(1, 3, 2)  # 1行2列，第2个子图
    # plt.title('Label(11th Frame)')
    # plt.imshow(img1, cmap='viridis', extent=[-128,127,-128,127])
    # # 添加颜色条
    # plt.colorbar()
    # plt.xlabel('X (Km)')
    #
    # # 绘制第3个图片（右侧子图）
    # plt.subplot(1, 3, 3)  # 1行3列，第3个子图
    # plt.title('Predict(11th Frame)')
    # plt.imshow(img2, cmap='viridis', extent=[-128,127,-128,127])
    # # plt.imshow(img2, cmap='gray')
    # # 添加颜色条
    # plt.colorbar()
    # plt.xlabel('X (Km)')
    #
    plt.figure(figsize=(10, 9))
    # # plt.figure(figsize=(15, 4))

    # 绘制第一个图片（左侧子图）
    plt.subplot(2, 2, 1)
    plt.title('$Z_{H}$')
    plt.imshow(img0, cmap='viridis', extent=[-128, 127, -128, 127])
    # plt.imshow(img0, cmap='gray')
    # # 添加颜色条
    plt.colorbar()

    # 添加横纵坐标轴标签
    # plt.xlabel('X (Km)')
    plt.ylabel('Y (Km)')

    # 绘制第二个图片（中间子图）
    plt.subplot(2, 2, 2)
    plt.title('$Z_{DR}$')
    plt.imshow(img3, cmap='viridis', extent=[-128, 127, -128, 127])
    # 添加颜色条
    plt.colorbar()
    # plt.xlabel('X (Km)')

    # 绘制第3个图片（右侧子图）
    plt.subplot(2, 2, 3)
    plt.title('Real Rain')
    plt.imshow(img1, cmap='viridis', extent=[-128, 127, -128, 127])
    # plt.imshow(img2, cmap='gray')
    # 添加颜色条
    plt.colorbar()
    plt.xlabel('X (Km)')
    plt.ylabel('Y (Km)')

    # 绘制第4个图片（右侧子图）
    plt.subplot(2, 2, 4)
    plt.title('Predict')
    plt.imshow(img2, cmap='viridis', extent=[-128, 127, -128, 127])
    # plt.imshow(img2, cmap='gray')
    # 添加颜色条
    plt.colorbar()
    plt.xlabel('X (Km)')


    # 保存重建图像
    plt.savefig(os.path.join(output_path, f"reconstructed_{i}.png"))

    plt.close()

def cal_loss(labels, predicts, log_file):


    loss_reconstruct = nn.MSELoss()
    # 定义损失函数为 SSIMLoss
    loss_ssim = SSIMLoss()

    # 初始化一个列表来存储每个通道的损失
    channel_losses_rec = []
    channel_losses_ss = []

    # 遍历每个通道
    for channel in range(predicts.size(1)):
        # 选择当前通道的模型输出和目标数据
        predict = predicts[:, channel, :, :]
        label = labels[:, channel, :, :]

        # 计算当前通道的损失
        loss_rec = loss_reconstruct(predict, label)
        # 计算整个批次中每个图像之间的 SSIM
        loss_ss = loss_ssim(label, predict)

        # 记录损失值
        channel_losses_rec.append(loss_rec.item())
        channel_losses_ss.append(loss_ss.item())

    if all(x > 0.0001 for x in channel_losses_rec) and all(x > 0.0001 for x in channel_losses_ss):
        log_message = f"loss_rec: {channel_losses_rec}\t\tloss_ss: {channel_losses_ss}"
        print(log_message)

        # 将信息写入日志文件
        log_file.write(log_message + '\n')
        log_file.flush()  # 确保信息被立即写入文件


def main():
    # 设置设备（CPU或GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")


    root_directory = '/home/kemove/data/datasets/f题/NJU_CPOL_update2308/dBZ/7.0km'
    root_directory1 = '/home/kemove/data/datasets/f题/NJU_CPOL_update2308/ZDR/7.0km'
    root_directory3 = '/home/kemove/data/datasets/f题/NJU_CPOL_update2308/KDP/7.0km'
    root_directory2 = '/home/kemove/data/datasets/f题/NJU_CPOL_kdpRain'

    # # 问题1用这个 DataSet
    # # 创建测试数据集和数据加载器
    # test_dataset = CustomDataset("/home/kemove/data/datasets/f题/NJU_CPOL_update2308/dBZ/3.0km", 10, 'dBZ')  # 替换为你的测试数据集路径
    # test_dataset = CustomDataset2(root_directory, root_directory1, root_directory3, num_samples=10)
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 问题3用这个DataSet
    test_dataset = CustomDataset1(root_directory, root_directory1, root_directory2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


    # # 问题1使用 30 通道输入，10通道输出
    # model = UNet(30, 10)
    # # 问题3使用 2 通道输入，1通道输出
    model = UNet(2, 1)

    model.load_state_dict(torch.load("/home/kemove/data/code/f/ckpt_threshold_3/parameter_980.pth"))
    # model.load_state_dict(torch.load("/home/kemove/data/code/f/ckpt_3/parameter_280.pth"))
    model.to(device)
    model.eval() 

    # 定义图像保存路径
    output_path = "../output_3/"

    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)

    # 进行图像重建
    with torch.no_grad():
        # 创建包含3个子图的图形
        # 保存打屏信息
        log_filename = os.path.join("/home/kemove/data/code/f/log/test.log")
        log_file = open(log_filename, 'w')

        for i, data in enumerate(test_loader):

            # plt.figure(figsize=(15, 4))

            inputs, labels = data 
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 将输入图像通过自动编码器进行重建
            predicts = model(inputs)

            # cal_loss(labels, predicts, log_file)
            if i < 20:
                savefig(inputs, labels, predicts, i, output_path)
            else:
                break

    print("Reconstruction completed. Reconstructed images saved in:", output_path)
    

if __name__ == '__main__':
    main()
