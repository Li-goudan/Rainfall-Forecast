import os
import argparse
import random

from torch.utils.data import DataLoader
from torch.utils.data import random_split

# from unet import UNet
from net.mynet import MyNet
from data.dataset import CustomDataset
from data.dataset import CustomDataset1
from data.dataset import CustomDataset2
from train import train


def get_command_line_parser():
    parser = argparse.ArgumentParser()

    # about dataset and network
    parser.add_argument('--project', type=str, default='/home/kemove/data/code/f')
    parser.add_argument('--log', type=str, default='log/training.log')
    parser.add_argument('--ckptpath', type=str, default='ckpt')
    parser.add_argument('--restore_path', type=str, default='ckpt/parameter_100.pth')
    parser.add_argument('--output_path', type=str, default='output')

    parser.add_argument('--epoch_max', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--log_freq', type=int, default=2)
    parser.add_argument('--save_freq', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--lamda_rec', type=float, default=1e+04)
    parser.add_argument('--lamda_ssim', type=float, default=1e+02)

    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=1024)

    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--restore', type=bool, default=False)
    parser.add_argument('--device', default='cuda')

    return parser



def main():
    parser = get_command_line_parser()
    args = parser.parse_args()
    random.seed(args.seed)

    root_directory = '/home/kemove/data/datasets/f题/NJU_CPOL_update2308/dBZ/1.0km'
    root_directory1 = '/home/kemove/data/datasets/f题/NJU_CPOL_update2308/ZDR/1.0km'
    root_directory3 = '/home/kemove/data/datasets/f题/NJU_CPOL_update2308/KDP/1.0km'
    root_directory2 = '/home/kemove/data/datasets/f题/NJU_CPOL_kdpRain'

    # 问题1的数据集（x,y）来自同一个文件夹
    # ds = CustomDataset('/home/kemove/data/datasets/f题/NJU_CPOL_update2308/dBZ/1.0km', num_samples=10, type='dBZ')
    ds = CustomDataset2(root_directory, root_directory1, root_directory3, num_samples=10)
    # 问题3的数据集（x,y）来自三个文件夹
    # ds = CustomDataset1(root_directory, root_directory1, root_directory2)
    
    n_train = int(len(ds) * 0.8)
    n_val = len(ds) - n_train

    ds_train, ds_val = random_split(ds, [n_train, n_val])

    dl_train = DataLoader(ds_train, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers,
                          drop_last=False)
    dl_val = DataLoader(ds_val, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False)

    # 问题1使用 10 通道输入，10通道输出
    # model = UNet(10, 10).to(args.device)
    model = UNet(30, 10).to(args.device)
    # 问题1使用 2 通道输入，1通道输出
    # model = UNet(2, 1).to(args.device)

    train(model, dl_train, dl_val, args)


if __name__ == '__main__':
    main()
