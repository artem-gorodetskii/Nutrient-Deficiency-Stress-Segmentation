# -*- coding: utf-8 -*-
# @Author: Artem Gorodetskii
# @Created Time: 3/22/2022 4:45 PM

from utils import seed_everything, DiceLoss, IoU, ValueWindow, adjust_learning_rate, print_log
from config import ModelConfig
from dataset import FieldsDataset
from model import SNUNet
from torch.utils.tensorboard import SummaryWriter
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import glob
import random
import torch.optim as optim
import numpy as np

# load model configuration
cfg = ModelConfig()

seed_everything(42)

# split data
folders = glob.glob(cfg.data_path + '/*')
random.shuffle(folders)
train_folders, valid_folders = folders[:round(len(folders)*(1-cfg.train_vaild_split_ratio))], folders[round(len(folders)*(1-cfg.train_vaild_split_ratio)):]


def train():
    """Train SNUNet model."""

    logs_dir = Path(cfg.logs_dir)
    writer = SummaryWriter(log_dir=cfg.logs_dir)
    logs_dir.mkdir(exist_ok=True)

    seed_everything(42)

    # creating datasets and dataloaders
    train_dataset = FieldsDataset(train_folders, augment=True, 
                                  size=cfg.size, normalize=True)
    train_loader = DataLoader(train_dataset, batch_size=cfg.BS, 
                              shuffle=True, num_workers=4, 
                              pin_memory=True, drop_last=True)

    valid_dataset = FieldsDataset(valid_folders, augment=False, size=cfg.size, normalize=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.BS, 
                              shuffle=True, num_workers=4, 
                              pin_memory=True, drop_last=True)

    model = SNUNet(input_channels=cfg.input_channels, norm_type=cfg.norm_type)

    if torch.cuda.is_available():
        model = model.cuda()
        device = torch.device('cuda')
        print("Device: CUDA\n")
    else:
        device = torch.device('cpu')
        print("Device: CPU\n")

    model.to(device)

    # regularization
    params = []
    for name, values in model.named_parameters():
        if 'coding' not in name and 'bias' not in name and 'proj' not in name and 'norm' not in name and 'fc' not in name and 'sSE' not in name:
            params += [{'params': [values], 'lr': cfg.initial_lr, 'weight_decay': cfg.weight_decay}]
        else:
            params += [{'params': [values], 'lr': cfg.initial_lr, 'weight_decay': 0.0}]

    optimizer = optim.Adam(params)

    dice = DiceLoss()
    bce = nn.BCELoss()
    iou = IoU()

    dice_window = ValueWindow(50)
    bce_window = ValueWindow(50)
    loss_window = ValueWindow(50)
    iou_window = ValueWindow(50)
    time_window = ValueWindow(50)

    model.train()

    # load backuo if necessary
    if cfg.load_backup:
      print('Weights loaded from: ' + str(cfg.loadpath))
      print('Continue training')
      model.load(cfg.loadpath, optimizer)

    print('Train')
    start_time = time.time()

    best_iou = 0.0
    # training
    for epoch in range(1, cfg.n_epochs + 1):
        for batch_idx, data in enumerate(train_loader):
            start_time_iter = time.time()

            imgs, masks, _ = data
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()

            masks_pred = model(imgs)
            step = model.get_step()

            dice_loss = dice(masks_pred, masks)
            bce_loss = bce(masks_pred.view(-1), masks.view(-1))
            loss = dice_loss + bce_loss
            loss.backward()

            with torch.no_grad():
                iou_value = iou((masks_pred>0.5).float(), masks)

            if cfg.grad_clip is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                           cfg.grad_clip,
                                                           norm_type=2.0)
                if np.isnan(grad_norm.cpu()):
                    print("grad_norm was NaN!")

            optimizer.step()
            current_lr = adjust_learning_rate(step, optimizer, cfg.initial_lr, cfg.gamma, cfg.milestones)

            dice_window.append(dice_loss.item())
            bce_window.append(bce_loss.item())
            loss_window.append(loss.item())
            iou_window.append(iou_value.item())
            time_window.append(time.time() - start_time_iter)

            average_dice_loss = dice_window.average
            average_bce_loss = bce_window.average
            average_loss = loss_window.average
            average_iou = iou_window.average
            average_time = time_window.average

            # print logs
            if step % cfg.log_every == 0:
                train_time = time.time()
                print_log(epoch, step, average_dice_loss, average_bce_loss,
                          average_loss, average_iou, start_time, train_time, average_time)
                
            writer.add_scalar('Train loss/DICE', average_dice_loss, step)
            writer.add_scalar('Train loss/BCE', average_bce_loss, step)
            writer.add_scalar('Train loss/Total', average_loss, step)
            writer.add_scalar('Train metrics/IOU', average_iou, step)

            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('LR', current_lr, step)

            # perform validation    
            if model.get_step() % cfg.test_every == 0:

                model.eval()
                with torch.no_grad():
                    val_dice_losses = []
                    val_bce_losses = []
                    val_losses = []
                    val_iou_metrics = []

                    print('Validation')

                    for val_batch_idx, val_data in enumerate(valid_loader):
                        val_imgs, val_masks, _ = val_data
                        val_imgs, val_masks = val_imgs.to(device), val_masks.to(device)

                        val_pred = model.generate(val_imgs)
                        val_step = model.get_step()

                        val_dice_loss = dice((val_pred>0.5).float(), val_masks)
                        val_bce_loss = bce(val_pred, val_masks)
                        val_loss = val_dice_loss + val_bce_loss
                        val_iou = iou((val_pred>0.5).float(), val_masks)

                        val_dice_losses.append(val_dice_loss.item())
                        val_bce_losses.append(val_bce_loss.item())
                        val_losses.append(val_loss.item())
                        val_iou_metrics.append(val_iou.item())

                    average_val_dice_loss = sum(val_dice_losses) / len(val_dice_losses)
                    average_val_bce_loss = sum(val_bce_losses) / len(val_bce_losses)
                    average_val_loss = sum(val_losses) / len(val_losses)
                    average_val_iou = sum(val_iou_metrics) / len(val_iou_metrics)

                    valid_time = time.time()
                    print_log(epoch, val_step, average_val_dice_loss, average_val_bce_loss,
                              average_val_loss, average_val_iou, start_time, valid_time, average_time)
                        
                    if average_val_iou >= best_iou:
                        best_iou = average_val_iou
                        model.save(cfg.savepath, optimizer)
                        print(f'Best step: {step}, IoU: {average_val_iou}')

                    writer.add_scalar('Test loss/DICE', average_val_dice_loss, step)
                    writer.add_scalar('Test loss/BCE', average_val_bce_loss, step)
                    writer.add_scalar('Test loss/Total', average_val_loss, step)
                    writer.add_scalar('Test metrics/IOU', average_val_iou, step)
                        
                model.train()


if __name__ == "__main__":
    train()
