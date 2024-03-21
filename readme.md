## 1.代码环境

* 操作系统：Ubuntu20.04
* 显卡：GeForce RTX 3060
* 显存：12G
* Python版本：3.8
* 网络框架：PyTorch

## 2. 项目文件结构

```
.
├── Q1&Q2					//针对问题1和2以及涉及问题3的代码，使用深度学习训练
│   ├── ckpt				//针对问题1、2、3训练得到的权重模型
│   │   ├── ckpt_1
│   │   ├── ckpt_2
│   │   └── ckpt_3
│   ├── data				//自定义数据载入实现
│   │   └── dataset.py
│   ├── log 	 			//训练及测试打屏信息记录
│   │   ├── test_1.log
│   │   ├── test_2.log
│   │   ├── training_1.log
│   │   ├── training_2.log
│   │   └── training_3.log
│   ├── net					//自定义网络实现
│   │   └── mynet.py
│   ├── output				//针对问题1、2、3的预测结果绘图输出
│   │   ├── output_1
│   │   ├── output_2
│   │   └── output_3
│   ├── read_npy			//数据可视化
│   │   ├── drawing
│   │   └── read_npy.py
│   └── src					//训练代码相关
│       ├── loss_ssim.py
│       ├── main.py
│       ├── test.py
│       └── train.py
├── Q3						//针对问题3的代码
│   ├── PhysicsFit.py
│   └── PhysicsRain.py
└── Q4						//针对问题4的代码
    └── ZH_ZDR.py
```

---

##自定义数据载入

```python
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


```

---

## 自定义网络结构

```python
class MyNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        x = x.view(-1, 30, 256, 256)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        out = torch.sigmoid(self.conv(dec1))
        # add mask
        mask = (out.abs() >= 0.05).float()
        out = out * mask

        return out

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
```

---

## 数据可视化

```python
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
```

---

## 构建 SSIM 损失

```python
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
```

---

## 训练脚本

```python
def train(model, dataloader_train, dataloader_val, args):
    # 保存打屏信息
    log_filename = os.path.join(args.project, args.log)
    log_file = open(log_filename, 'w')


    torch.manual_seed(args.seed)
    loss_reconstruct = nn.MSELoss()
    # 定义损失函数为 SSIMLoss
    loss_ssim = SSIMLoss()

    # 超参
    epoch_max = args.epoch_max
    learning_rate = args.learning_rate
    log_freq = args.log_freq
    save_freq = args.save_freq
    device = args.device


    # restore
    if args.restore:
        checkpoint = torch.load(os.path.join(args.project, args.restore_path))
        model.load_state_dict(checkpoint)
        print("============   Restore   ==============")

    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("============Start Training==============")
    nowtime = datetime.datetime.now()
    print("=====Time: {}======".format(nowtime))

    for epoch in range(epoch_max):
        start_time = time.time()
        torch.cuda.empty_cache()
        torch.autograd.set_detect_anomaly(True)
        loss_sum = 0.0
        loss_rec_sum = 0.0
        loss_ss_sum = 0.0
        step_max = len(dataloader_train)

        model.train()
        for step, (inputs, labels) in enumerate(dataloader_train):
            inputs = inputs.to(device)
            labels = labels.to(device)
            predicts = model(inputs)

            # loss_rec = loss_func(predicts, labels)
            loss_rec = loss_reconstruct(predicts, labels)
            # print(loss_rec)
            # 计算整个批次中每个图像之间的 SSIM
            loss_ss = loss_ssim(labels, predicts)

            loss = args.lamda_rec * loss_rec + args.lamda_ssim * loss_ss
            # loss = args.lamda_rec * loss_rec
            # loss = args.lamda_rec * loss_rec + args.lamda_l2 * loss_MSE


            optim.zero_grad()
            loss.backward(retain_graph=True)
            optim.step()
            # # 打印threshold参数的值
            # print("Threshold value:", model.threshold.item())

            loss_sum += loss.item()
            loss_rec_sum += loss_rec.item()
            loss_ss_sum += loss_ss.item()

        end_time = time.time()

        if epoch % log_freq == 0 and step == step_max - 1:
            time_cost = end_time - start_time

            # 打印信息并记录到日志文件
            # log_message = f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}"
            log_message = f"[Epoch = {epoch}/{epoch_max}]\t\tloss: {(loss_sum / step_max):.4f}\t\tloss_rec: {(loss_rec_sum/step_max):.4f}\t\tloss_ss: {(loss_ss_sum/step_max):.4f}\t\ttime: {time_cost:.4f}"
            print(log_message)

            # 将信息写入日志文件
            log_file.write(log_message + '\n')
            log_file.flush()

            # print("[Epoch = {}/{}]\t\tloss: {:.4f}\t\tloss_rec: {:.4f}\t\tloss_ss: {:.4f}\t\ttime: {:.4f}".format(epoch, epoch_max,
            #                                                                          loss_sum / step_max, loss_rec_sum/step_max, loss_ss_sum/step_max, time_cost))

        if epoch % save_freq == 0:
            loss_sum = 0
            loss_rec_sum = 0.0

            model.eval()
            with torch.no_grad():
                for step, (inputs, labels) in enumerate(dataloader_val):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    predicts = model(inputs)

                    loss_rec = loss_reconstruct(predicts, labels)
                    # 计算整个批次中每个图像之间的 SSIM
                    loss_ss = loss_ssim(labels, predicts)

                    loss = args.lamda_rec * loss_rec + args.lamda_ssim * loss_ss

                    loss_sum += loss.item()
                    loss_rec_sum += loss_rec.item()
                    loss_ss_sum += loss_ss.item()

            print("[Epoch = {}/{}]\t\tloss: {:.4f}\t\tloss_rec: {:.4f}\t\tloss_ss: {:.4f}".format(epoch, epoch_max,
                                                                                     loss_sum / step_max, loss_rec_sum/step_max, loss_ss_sum/step_max))

            torch.save(model.state_dict(), os.path.join(args.project, args.ckptpath, "parameter_{}.pth").format(epoch))
```

---

## 测试脚本

```python
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
```

---

## 降雨关系拟合

```python
import os
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
import random
import cv2
import matplotlib.pyplot as plt

# 文件夹路径
dbz_folder = "/media/kemove/9c9d1ba8-bdd4-4a5f-a593-3de9968e5669/Math_Ladar/NJU_CPOL_update2308/dBZ/1.0km/"
zdr_folder = "/media/kemove/9c9d1ba8-bdd4-4a5f-a593-3de9968e5669/Math_Ladar/NJU_CPOL_update2308/ZDR/1.0km/"
kdp_folder = "/media/kemove/9c9d1ba8-bdd4-4a5f-a593-3de9968e5669/Math_Ladar/NJU_CPOL_update2308/KDP/1.0km/"
rain_folder_base = "/media/kemove/9c9d1ba8-bdd4-4a5f-a593-3de9968e5669/Math_Ladar/NJU_CPOL_kdpRain/"

# 由于Rain文件夹下有data_dir_000到data_dir_257，我们将循环遍历处理每个文件夹
num_rain_dirs = 2  # data_dir_000到xxx

# 初始化存储数据的列表
dbz_values_list = []
zdr_values_list = []
kdp_values_list = []
rain_values_list = []


def sample_data(data, num_samples):
    # 获取 rain_data 的形状
    rain_shape = data.shape[:2]

    # 找到 rain_data 非零值的索引
    nonzero_indices = np.argwhere(data[:, :, 2] > 0)

    # 初始化采样点列表
    sampled_points = []

    # 遍历每个非零点
    for index in nonzero_indices:
        i, j = index

        # 检查该点周围的非零点数量
        non_zero_neighbor_count = 0

        for ni in range(i - 1, i + 2):
            for nj in range(j - 1, j + 2):
                if 0 <= ni < rain_shape[0] and 0 <= nj < rain_shape[1] and data[ni, nj, 2] > 0:
                    non_zero_neighbor_count += 1

            # 如果邻居点非零的数量满足要求，则将该点加入采样列表
        if non_zero_neighbor_count >= 4:
            sampled_points.append(data[i, j])

    # 随机选择100个样本
    num_available_samples = len(sampled_points)
    num_samples_to_select = min(num_samples, num_available_samples)
    sampled_indices = random.sample(range(num_available_samples), num_samples_to_select)

    # 获取选定的样本
    sampled_data = np.array([sampled_points[i] for i in sampled_indices])
    return sampled_data


def feature_data(data, keep_percent):
    # 使用ORB算法
    rain_data = data[:, :, 3]

    data2_uint8 = rain_data.astype(np.uint8)
    gray_image = cv2.cvtColor(data2_uint8, cv2.COLOR_GRAY2BGR)

    detection_method = cv2.ORB_create()
    keypoints = detection_method.detect(gray_image, None)

    keypoints = sorted(keypoints, key=lambda x: -x.response)

    # 计算保留的特征点数量（前20%）
    keep_percent = keep_percent
    num_to_keep = int(keep_percent * len(keypoints))
    selected_keypoints = keypoints[:num_to_keep]
    result_image = cv2.drawKeypoints(gray_image, selected_keypoints, None, color=(0, 255, 0))

    # 保存特征点坐标
    feature_points = []
    for kp in selected_keypoints:
        x, y = kp.pt
        x_rounded = round(x)
        y_rounded = round(y)
        # 交换x和y的位置

        feature_points.append((int(y_rounded), int(x_rounded)))
    # 根据特征点坐标获取对应的数据
    extracted_data = np.zeros((len(feature_points), 4))  # 初始化数组
    for i, (x, y) in enumerate(feature_points):
        extracted_data[i] = data[x, y]
        # print(f"特征点坐标：({x}, {y})")
    return extracted_data
    # 画图
    # scale_percent = 300  # 调整的百分比
    # width = int(result_image.shape[1] * scale_percent / 100)
    # height = int(result_image.shape[0] * scale_percent / 100)
    # dim = (width, height)
    # result_image_resized = cv2.resize(result_image, dim, interpolation=cv2.INTER_AREA)
    #
    # # 显示带有特征点的调整后的图像
    # cv2.imshow('FAST Feature Detection', result_image_resized)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


# 定义损失函数，这里使用平方损失
def loss_function_three(params, dbz_values, zdr_values, rain_values):
    y_predicted = fit_function_three(params, dbz_values, zdr_values)
    return y_predicted - rain_values

def loss_function_two(params, dbz_values, rain_values):
    y_predicted = fit_function_two(params, dbz_values)
    return y_predicted - rain_values

def loss_function_two_KDP(params, kdp_values, rain_values):
    y_predicted = fit_function_two_KDP(params, kdp_values)
    return y_predicted - rain_values

def loss_function_three_KDP(params, zdr_values, kdp_values, rain_values):
    y_predicted = fit_function_three_KDP(params, zdr_values, kdp_values)
    return y_predicted - rain_values

def loss_function_two_ZDR(params, zdr_values, rain_values):
    y_predicted = fit_function_two_ZDR(params, zdr_values)
    return y_predicted - rain_values


# 重新定义拟合模型函数，参数打包成一个数组
def fit_function_three(params, dbz_values, zdr_values):
    a, b, c = params
    return a * (dbz_values ** b) * (zdr_values ** c)

def fit_function_two(params, dbz_values):
    a, b = params
    return a * (dbz_values ** b)

def fit_function_two_KDP(params, kdp_values):
    a, b = params
    return a * (kdp_values ** b)

def fit_function_three_KDP(params, zdr_values, kdp_values):
    a, b, c = params
    return a * (zdr_values ** b) * (kdp_values ** c)

def fit_function_two_ZDR(params, zdr_values):
    a, b = params
    return a * (zdr_values ** b)

# def fit_function(data, a, b, c):
#     # 幂函数拟合形式: Rain = a × (dBZ^b) × (ZDR^c)
#     dbz_values, zdr_values = data
#     return a * (dbz_values ** b) * (zdr_values ** c)



# 计算拟合参数函数
def calculate_parameters(dbz_values, zdr_values, kdp_values, rain_values):
    # # curve_fit的方式
    # popt, _ = curve_fit(fit_function, (dbz_values, zdr_values), rain_values, maxfev=5000)
    # # 从拟合结果中获取拟合参数
    # a3_fit, b3_fit, c3_fit = popt

    # 以下使用 least_squares 进行拟合
    initial_params_two = [0.04722 , 0.6346]
    resulttwo = least_squares(loss_function_two, initial_params_two, args=(dbz_values, rain_values))
    a2_fit, b2_fit = resulttwo.x

    initial_params_three = [0.003436, 0.93485, -0.75121 ]
    resultthree = least_squares(loss_function_three, initial_params_three, args=(dbz_values, zdr_values, rain_values))
    a3_fit, b3_fit, c3_fit = resultthree.x

    initial_params_two_KDP = [26.81 , 0.7224]
    resulttwo_KDP = least_squares(loss_function_two_KDP, initial_params_two_KDP, args=(kdp_values, rain_values))
    a2_KDP_fit, b2_KDP_fit = resulttwo_KDP.x

    initial_params_three_KDP = [32.2686, -0.7487, 1.0181]
    resultthree_KDP = least_squares(loss_function_three_KDP, initial_params_three_KDP, args=(zdr_values, kdp_values, rain_values))
    a3_KDP_fit, b3_KDP_fit, c3_KDP_fit = resultthree_KDP.x

    initial_params_two_ZDR = [26.81, 0.7224]
    resulttwo_ZDR = least_squares(loss_function_two_ZDR, initial_params_two_ZDR, args=(zdr_values, rain_values))
    a2_ZDR_fit, b2_ZDR_fit = resulttwo_ZDR.x

    return a3_fit, b3_fit, c3_fit, a2_fit, b2_fit, a2_KDP_fit, b2_KDP_fit, a3_KDP_fit, b3_KDP_fit, c3_KDP_fit, a2_ZDR_fit, b2_ZDR_fit

if __name__ == "__main__":
    # 遍历dBZ、ZDR、Rain文件夹
    for i in range(num_rain_dirs):
        # 构建Rain文件夹路径
        rain_folder_path = os.path.join(rain_folder_base, f"data_dir_{i:03d}")

        # 遍历对应的dBZ、ZDR文件夹
        dbz_folder_path = os.path.join(dbz_folder, f"data_dir_{i:03d}")
        zdr_folder_path = os.path.join(zdr_folder, f"data_dir_{i:03d}")
        kdp_folder_path = os.path.join(kdp_folder, f"data_dir_{i:03d}")

        # 遍历npy文件
        for file_name in os.listdir(rain_folder_path):
            if file_name.endswith(".npy"):
                rain_file_path = os.path.join(rain_folder_path, file_name)
                dbz_file_path = os.path.join(dbz_folder_path, file_name)
                zdr_file_path = os.path.join(zdr_folder_path, file_name)
                kdp_file_path = os.path.join(kdp_folder_path, file_name)

                # 读取npy文件
                rain_data = np.load(rain_file_path)
                dbz_data = np.load(dbz_file_path)
                zdr_data = np.load(zdr_file_path)
                kdp_data = np.load(kdp_file_path)

                # 对dBZ和ZDR进行转换
                dbz_data = 10 ** (0.1 * dbz_data)
                zdr_data = 10 ** (0.1 * zdr_data)
                kdp_data = 10 ** (0.1 * kdp_data)

                # rain_data = 10*np.log(rain_data)

                # 创建一个三维矩阵，每个采样点对应三个数据：dbz_data, zdr_data, rain_data
                sampled_data = np.stack((dbz_data, zdr_data, kdp_data, rain_data))

                # 调整维度使其符合采样函数的输入要求
                sampled_data = np.moveaxis(sampled_data, 0, -1)

                # 采样特征点
                sampled_data = feature_data(sampled_data, keep_percent=0.2)
                # 采样数据
                # sampled_data = sample_data(sampled_data, 20)

                # 遍历每个采样点
                for point in sampled_data:
                    # 采样点的dBZ、ZDR、Rain值分别存入对应的列表
                    dbz_values_list.append(point[0])  # dBZ值
                    zdr_values_list.append(point[1])  # ZDR值
                    kdp_values_list.append(point[2])  # KDP值
                    rain_values_list.append(point[3])  # Rain值


    # 将列表转换为numpy数组
    dbz_values = np.array(dbz_values_list)
    zdr_values = np.array(zdr_values_list)
    kdp_values = np.array(kdp_values_list)
    rain_values = np.array(rain_values_list)

    # 计算Rain = a × dBZ^b × ZDR^c
    a3_fit, b3_fit, c3_fit, a2_fit, b2_fit, a2_KDP_fit, b2_KDP_fit, a3_KDP_fit, b3_KDP_fit, c3_KDP_fit, a2_ZDR_fit, b2_ZDR_fit = calculate_parameters(dbz_values, zdr_values, kdp_values, rain_values)

    # 输出a、b、c的值
    print("Parameters_two: a2_fit_ =", a2_fit, ", b2_fit_ =", b2_fit)
    print("Parameters_three: a3_fit_ =", a3_fit, ", b3_fit_ =", b3_fit, ", c3_fit_ =", c3_fit)
    print("Parameters_two_with_KDP: a2_KDP_fit_ =", a2_KDP_fit, ", b2_KDP_fit_ =", b2_KDP_fit)
    print("Parameters_three_with_KDP: a3_KDP_fit_ =", a3_KDP_fit, ", b3_KDP_fit_ =", b3_KDP_fit, ", c3_KDP_fit_ =", c3_KDP_fit)
    print("Parameters_two_with_ZDR: a2_ZDR_ZDR_ =", a2_ZDR_fit, ", b2_ZDR_fit_ =", b2_ZDR_fit)
```

---

## 降雨预测

```python
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import mean_squared_error, mean_absolute_error
num_rain_dirs = 2
# 拟合参数
#最优
# least_squares——————————————
a3_fit_ = 0.003436
b3_fit_ = 0.93485
c3_fit_ = -0.75121
# least_squares——————————————
a2_fit_ = 0.04722
b2_fit_ = 0.6346

#计算
# Parameters_two——————————————
# a2_fit_ = 44.94031533753499
# b2_fit_ = 0.05093532406125978
# # Parameters_three——————————————
# a3_fit_ = 45.4915289319782
# b3_fit_ = 0.045675847822630206
# c3_fit_ = 0.28262395781421634
# Parameters_two_with_KDP——————————————
a2_KDP_fit_ = 36.49580920159213
b2_KDP_fit_ = 1.7046392651466873
# Parameters_three_with_KDP——————————————
a3_KDP_fit_ = 36.30905791568315
b3_KDP_fit_ = -0.05990652067810124
c3_KDP_fit_ = 1.7294407894120694
#Parameters_two_with_ZDR——————————————
a2_ZDR_ZDR_ = 50.022515568245126
b2_ZDR_fit_ = 0.14785741499811608

# Parameters_two: a = 78.16592584439839 , b = 0.04714547555161856
# Parameters_three: a = 75.18704918447165 , b = 0.05660787727053795 , c = -0.19022797169935685
# Parameters_two_with_KDP: a = 83.65228184773535 , b = 0.41205386568466407
# Parameters_three_with_KDP: a = 82.0646202591343 , b = 0.14428451341655285 , c = 0.4263167475782254

# 文件夹路径
dbz_folder_base = "/media/kemove/9c9d1ba8-bdd4-4a5f-a593-3de9968e5669/Math_Ladar/NJU_CPOL_update2308/dBZ/1.0km/"
zdr_folder_base = "/media/kemove/9c9d1ba8-bdd4-4a5f-a593-3de9968e5669/Math_Ladar/NJU_CPOL_update2308/ZDR/1.0km/"
kdp_folder_base = "/media/kemove/9c9d1ba8-bdd4-4a5f-a593-3de9968e5669/Math_Ladar/NJU_CPOL_update2308/KDP/1.0km/"
rain_folder_base = "/media/kemove/9c9d1ba8-bdd4-4a5f-a593-3de9968e5669/Math_Ladar/NJU_CPOL_kdpRain/"
phy_pre_kdpRain_base = "/media/kemove/9c9d1ba8-bdd4-4a5f-a593-3de9968e5669/Math_Ladar/Phy_pre_kdpRain"

# 初始化存储真实Rain数据和预测Rain数据的列表
real_rain_values = []
predicted_rain_values = []

def eval(real_rain_data, predicted_rain):
    # 将 real_rain_data 的最大最小值间分为4个区间
    num_bins = 4
    bin_edges = np.linspace(real_rain_data.min(), real_rain_data.max(), num_bins + 1)

    # 初始化结果存储列表
    bin_counts = []
    correlation_values = []
    std_errors = []
    rmse_values = []
    mae_values = []
    max_values = []

    # 控制小数点后三位的格式
    float_format = "{:.3f}"

    for i in range(num_bins):
        # 选择位于当前区间的数据点
        mask = np.logical_and(real_rain_data >= bin_edges[i], real_rain_data < bin_edges[i + 1])
        points_in_bin = predicted_rain[mask]

        # 计算当前区间的数据点数量
        bin_counts.append(np.sum(mask))

        # 计算当前区间与 predicted_rain 的相关系数
        correlation = np.corrcoef(points_in_bin.flatten(), real_rain_data[mask].flatten())[0, 1]
        correlation_values.append(correlation)

        # 计算分数标准误差、均方根误差和平均绝对误差
        std_error = np.std(points_in_bin.flatten())
        std_errors.append(std_error)

        rmse = np.sqrt(mean_squared_error(real_rain_data[mask], points_in_bin.flatten()))
        rmse_values.append(rmse)

        mae = mean_absolute_error(real_rain_data[mask], points_in_bin.flatten())
        mae_values.append(mae)

        # 计算当前区间的最大值
        max_value = np.max(real_rain_data[mask])
        max_values.append(max_value)

    # 打印结果
    for i in range(num_bins):
        print(
            f"区间 {i + 1}(最大值 = {float_format.format(max_values[i])}): 数据点数量 = {bin_counts[i]}, 相关系数 = {float_format.format(correlation_values[i])}, " \
            f"分数标准误差 = {float_format.format(std_errors[i])}, 均方根误差 = {float_format.format(rmse_values[i])}, 平均绝对误差 = {float_format.format(mae_values[i])}")

def plot_eval(real_rain_data, predicted_rain, diff, correlation):
    # 画图对比real_rain_data、predicted_rain和差别
    plt.figure(figsize=(10, 10))
    image_size = len(real_rain_data)

    # 计算图像中心坐标
    center_x, center_y = image_size / 2, image_size / 2

    # 计算雷达图的半径
    max_radius = center_x
    num_circles = 5
    circle_radii = np.linspace(0, max_radius, num_circles)

    # 绘制12条射线
    num_lines = 12
    angle_step = 2 * np.pi / num_lines

    colors = [

        '#D8E8E5',
        '#00CED1',
        '#5499C7',
        '#C2B5C7',
        '#929BC2',
        '#72A373',
        '#FFCDD2',
        '#C8686C',
        '#FFA500',
        '#FFECB3',
        '#FFD700',
        '#FFFF00'
    ]
    cmap = ListedColormap(colors)
    # 绘制实际降雨图像
    plt.subplot(2, 2, 1)
    plt.imshow(real_rain_data, cmap=cmap, extent=[-center_x, center_x, -center_y, center_y])
    plt.title('Real Rain')
    plt.colorbar(shrink=0.75)
    plt.xlabel('X (km)')
    plt.ylabel('Y (km)')

    # 绘制雷达图
    for radius in circle_radii:
        circle = plt.Circle((0, 0), radius, color='white', linestyle='-', fill=False, linewidth=1, alpha=0.7)
        plt.gca().add_patch(circle)

    for i in range(num_lines):
        angle = i * angle_step
        x_end = max_radius * np.cos(angle)
        y_end = max_radius * np.sin(angle)
        plt.plot([0, x_end], [0, y_end], 'white', linestyle='-', linewidth=1, alpha=0.7)

    # 绘制预测降雨图像
    plt.subplot(2, 2, 2)
    plt.imshow(predicted_rain, cmap=cmap, extent=[-center_x, center_x, -center_y, center_y])
    plt.title('Predicted Rain')
    plt.colorbar(shrink=0.75)
    plt.xlabel('X (km)')
    plt.ylabel('Y (km)')

    # 绘制雷达图
    for radius in circle_radii:
        circle = plt.Circle((0, 0), radius, color='white', linestyle='-', fill=False, linewidth=1, alpha=0.7)
        plt.gca().add_patch(circle)

    for i in range(num_lines):
        angle = i * angle_step
        x_end = max_radius * np.cos(angle)
        y_end = max_radius * np.sin(angle)
        plt.plot([0, x_end], [0, y_end], 'white', linestyle='-', linewidth=1, alpha=0.7)

    # 绘制差别图像
    plt.subplot(2, 2, 3)
    plt.imshow(diff, cmap=cmap, extent=[-center_x, center_x, -center_y, center_y])
    plt.title('Difference (Real - Predicted)')
    plt.colorbar(shrink=0.75)
    plt.xlabel('X (km)')
    plt.ylabel('Y (km)')

    # 绘制雷达图
    for radius in circle_radii:
        circle = plt.Circle((0, 0), radius, color='white', linestyle='-', fill=False, linewidth=1, alpha=0.7)
        plt.gca().add_patch(circle)

    for i in range(num_lines):
        angle = i * angle_step
        x_end = max_radius * np.cos(angle)
        y_end = max_radius * np.sin(angle)
        plt.plot([0, x_end], [0, y_end], 'white', linestyle='-', linewidth=1, alpha=0.7)

    # 绘制拟合散点图
    plt.subplot(2, 2, 4)
    plt.scatter(real_rain_data.flatten(), predicted_rain.flatten(), alpha=0.5, c='#6587B5')
    plt.xlabel('Real Rain')
    plt.ylabel('Predicted Rain')
    plt.title('Correlation')

    plt.text(0.5, 0.5, 'Correlation: {:.2f}'.format(correlation),
             horizontalalignment='center',
             verticalalignment='center',
             fontsize=12,
             transform=plt.gca().transAxes)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 循环处理接下来的文件夹
    for i in range(num_rain_dirs, num_rain_dirs + 257):
        # 构建文件夹路径
        dbz_folder_path = os.path.join(dbz_folder_base, f"data_dir_{i:03d}")
        zdr_folder_path = os.path.join(zdr_folder_base, f"data_dir_{i:03d}")
        kdp_folder_path = os.path.join(kdp_folder_base, f"data_dir_{i:03d}")
        phy_pre_kdpRain_path = os.path.join(phy_pre_kdpRain_base, f"data_dir_{i:03d}")

        # 创建对应的Phy_pre_kdpRain文件夹
        os.makedirs(phy_pre_kdpRain_path, exist_ok=True)

        # 遍历npy文件，按文件名数字顺序排序
        # file_names = sorted(os.listdir(dbz_folder_path), key=lambda x: int(x.split('_')[2].split('.')[0]))

        # 初始化存储相关系数的列表
        correlation_coefficients = []

        # 遍历npy文件
        for file_name in os.listdir(dbz_folder_path):
            if file_name.endswith(".npy"):
                dbz_file_path = os.path.join(dbz_folder_path, file_name)
                zdr_file_path = os.path.join(zdr_folder_path, file_name)
                kdp_file_path = os.path.join(kdp_folder_path, file_name)

                # 读取npy文件
                dbz_data = np.load(dbz_file_path)
                zdr_data = np.load(zdr_file_path)
                kdp_data = np.load(kdp_file_path)

                # 对dBZ和ZDR进行转换
                dbz_data = 10 ** (0.1 * dbz_data)
                zdr_data = 10 ** (0.1 * zdr_data)
                kdp_data = 10 ** (0.1 * kdp_data)

                # 读取真实的Rain值
                # 这里假设真实Rain值也是从相同文件夹读取，文件名相同只是存放路径不同，实际情况请根据您的数据结构调整
                real_rain_file_path = os.path.join(rain_folder_base, f"data_dir_{i:03d}", file_name)
                real_rain_data = np.load(real_rain_file_path)

                mask = np.zeros_like(real_rain_data)
                mask[real_rain_data > 0] = 1


                dbz_data = dbz_data * mask
                zdr_data = zdr_data * mask
                kdp_data = kdp_data * mask

                # 俩参数预测
                # predicted_rain = a2_fit_ * (dbz_data ** b2_fit_) - 40  # 将 nan 项和小于 0 项设置为 0前的修正
                # predicted_rain = 3.3*predicted_rain

                # 三参数预测
                predicted_rain = a3_fit_ * (dbz_data ** b3_fit_) * (zdr_data ** c3_fit_) - 48
                predicted_rain = 3.3*predicted_rain

                # 俩参数预测-带KDP
                # predicted_rain = a2_KDP_fit_ * (kdp_data ** b2_KDP_fit_) - 30
                # predicted_rain = 1.3 * predicted_rain

                # 三参数预测-带KDP
                # predicted_rain = a3_KDP_fit_ * (zdr_data ** b3_KDP_fit_) * (kdp_data ** c3_KDP_fit_) - 30

                # 俩参数预测-带ZDR
                predicted_rain = a2_ZDR_ZDR_ * (zdr_data ** b2_ZDR_fit_) - 50
                predicted_rain = 1.3 * predicted_rain

                # 将 nan 项和小于 0 项设置为 0
                predicted_rain[np.isnan(predicted_rain) | (predicted_rain < 0)] = 0

                #----以下绘归一化绝对误差图代码----

                #已有dbz_data、zdr_data、real_rain_data、计算出的predicted_rain

                # ----以上绘归一化绝对误差图代码----

                #计算real_rain_data和predicted_rain的差别
                diff = real_rain_data - predicted_rain

                # 计算相关系数
                correlation = np.corrcoef(real_rain_data.flatten(), predicted_rain.flatten())[0, 1]
                correlation_coefficients.append(correlation)

                eval(real_rain_data, predicted_rain)

                plot_eval(real_rain_data, predicted_rain, diff, correlation)



                # 存储predicted_rain数据为.npy
                # predicted_rain_file_path = os.path.join(phy_pre_kdpRain_path, file_name)
                # np.save(predicted_rain_file_path, predicted_rain)
```

---

## 误差分析

```python
import os
import numpy as np
import matplotlib.pyplot as plt
# from PhysicsFit import num_rain_dirs
from matplotlib.colors import ListedColormap
from sklearn.metrics import mean_squared_error, mean_absolute_error
import cv2


# Parameters_two——————————————
a2_fit_ = 44.94031533753499
b2_fit_ = 0.05093532406125978
# Parameters_three——————————————
a3_fit_ = 45.4915289319782
b3_fit_ = 0.045675847822630206
c3_fit_ = 0.28262395781421634
# Parameters_two_with_KDP——————————————
a2_KDP_fit_ = 36.49580920159213
b2_KDP_fit_ = 1.7046392651466873
# Parameters_three_with_KDP——————————————
a3_KDP_fit_ = 36.30905791568315
b3_KDP_fit_ = -0.05990652067810124
c3_KDP_fit_ = 1.7294407894120694


data = np.loadtxt('sampled_data_all_more.txt')

# for a in range(4):
#     if a == 0:
#         predicted_rain = a2_fit_ * (data[:, 0] ** b2_fit_) - 50
#         predicted_rain = 3.3*predicted_rain
#     if a == 1:
#         predicted_rain = a3_fit_ * (data[:, 0] ** b3_fit_) * (data[:, 1] ** c3_fit_) - 48
#         predicted_rain = 3.3*predicted_rain
#     if a == 2:
#         predicted_rain = a2_KDP_fit_ * (data[:, 2] ** b2_KDP_fit_) - 30
#         predicted_rain = 1.3 * predicted_rain
#     if a == 3:
#         predicted_rain = a3_KDP_fit_ * (data[:, 1] ** b3_KDP_fit_) * (data[:, 2] ** c3_KDP_fit_) - 30

a = 4

# predicted_rain = a2_fit_ * (data[:, 0] ** b2_fit_) - 50
# predicted_rain = 3.3*predicted_rain

# predicted_rain = a3_fit_ * (data[:, 0] ** b3_fit_) * (data[:, 1] ** c3_fit_) - 48
# predicted_rain = 3.3*predicted_rain

# predicted_rain = a2_KDP_fit_ * (data[:, 2] ** b2_KDP_fit_) - 30
# predicted_rain = 1.3 * predicted_rain
#
predicted_rain = a3_KDP_fit_ * (data[:, 1] ** b3_KDP_fit_) * (data[:, 2] ** c3_KDP_fit_) - 30
#
data[:, :3] = 10 * np.log10(data[:, :3])


ne_rain = np.abs(predicted_rain - data[:, 3])
data[:, 3] = ne_rain

sorted_indices = np.argsort(data[:, 0])
sorted_data = data[sorted_indices]
sorted_data = sorted_data[sorted_data[:, 0] > 0]

map = np.zeros((32, 32))
for i in np.arange(0, 64, 2):
    filtered_data = sorted_data[(sorted_data[:, 0] >= i) & (sorted_data[:, 0] < i+2)]
    for j in np.arange(0, 5, 0.2):
        if filtered_data is not None:
            filtered_data_zdr = filtered_data[(filtered_data[:, 1] >= j) & (filtered_data[:, 1] < j+0.2)]
            value = np.sum(filtered_data_zdr[:, 3])/np.size(filtered_data_zdr[:, 3])
            map[int(31-j*5), int(i/2)] = value

nan_indices = np.isnan(map)

# 将NaN值替换为0
map[nan_indices] = 0
map[map > 300] = 250

plt.imshow(map, cmap= 'coolwarm', interpolation='none')  # viridis是一种颜色映射(colormap)，可以根据需要选择其他的colormap
plt.colorbar()  # 添加颜色条，显示数值与颜色的对应关系
for i in range(1, map.shape[1]):
    plt.axhline(i - 1, color='white', linewidth=0.5, linestyle='dashed')
    plt.axvline(i - 1, color='white', linewidth=0.5, linestyle='dashed')
plt.savefig('{}.png'.format(a), dpi=300)
```

