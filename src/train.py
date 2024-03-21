import os
import time
import datetime
import torch
from torch import nn
from loss_ssim import SSIMLoss


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
