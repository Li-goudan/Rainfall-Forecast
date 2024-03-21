#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
    @Project ：f 
    @File    ：loss_ssim.py
    @IDE     ：PyCharm 
    @Author  ：kemove
    @Date    ：2023/9/23 上午12:05 
"""
import torch
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim

class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()

    def forward(self, x, y):
        # 计算 SSIM
        ssim_value = torch.zeros(x.size(0), device=x.device, dtype=torch.float32)

        for i in range(x.size(0)):
            # 问题1尺寸为 10*256*256，计算 ssim 需要加 channel_axis=-3 以计算多通道
            # ssim_val = ssim(x[i].squeeze().cpu().detach().numpy(), y[i].squeeze().detach().cpu().numpy(), data_range=1.0, channel_axis=-3)
            ssim_val = ssim(x[i].squeeze().cpu().detach().numpy(), y[i].squeeze().detach().cpu().numpy(), data_range=1.0)
            ssim_value[i] = torch.tensor(ssim_val, dtype=torch.float32, device=x.device)
        # 计算平均 SSIM
        mean_ssim = torch.mean(ssim_value)

        # 返回 1 - SSIM 作为损失，使得损失越小越好
        loss = 1.0 - mean_ssim

        return loss
