## 1.代码环境

* 操作系统：Ubuntu20.04
* 显卡：GeForce RTX 3060
* 显存：12G
* Python版本：3.8
* 网络框架：PyTorch

## 2. 项目文件结构

```
.
└── RainfallForecast			//针对问题1和2以及涉及问题3的代码，使用深度学习训练
    ├── ckpt				//针对问题1、2、3训练得到的权重模型
    │   ├── ckpt_1
    │   ├── ckpt_2
    │   └── ckpt_3
    ├── data				//自定义数据载入实现
    │   └── dataset.py
    ├── log 	 			//训练及测试打屏信息记录
    │   ├── test_1.log
    │   ├── test_2.log
    │   ├── training_1.log
    │   ├── training_2.log
    │   └── training_3.log
    ├── net				//自定义网络实现
    │   └── mynet.py
    ├── output				//针对问题1、2、3的预测结果绘图输出
    │   ├── output_1
    │   ├── output_2
    │   └── output_3
    ├── read_npy			//数据可视化
    │   ├── drawing
    │   └── read_npy.py
    └── src				//训练代码相关
        ├── loss_ssim.py
        ├── main.py
        ├── test.py
        └── train.py

```

---

## 自定义数据载入

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
