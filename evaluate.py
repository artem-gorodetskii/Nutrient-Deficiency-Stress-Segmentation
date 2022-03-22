# -*- coding: utf-8 -*-
# @Author: Artem Gorodetskii
# @Created Time: 3/22/2022 4:45 PM

from utils import seed_everything, DiceLoss, IoU
from config import ModelConfig
from dataset import FieldsDataset
from model import SNUNet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import glob
import random
import torch.optim as optim
import numpy as np


seed_everything(42)

# load model configuration
cfg = ModelConfig()

# split data
folders = glob.glob(cfg.data_path + '/*')
random.shuffle(folders)
train_folders, valid_folders = folders[:round(len(folders)*(1-cfg.train_vaild_split_ratio))], folders[round(len(folders)*(1-cfg.train_vaild_split_ratio)):]


model = SNUNet(input_channels=cfg.input_channels, norm_type=cfg.norm_type)

params = []
for name, values in model.named_parameters():
    if 'coding' not in name and 'bias' not in name and 'proj' not in name and 'norm' not in name and 'fc' not in name and 'sSE' not in name:
        params += [{'params': [values], 'lr': cfg.initial_lr, 'weight_decay': cfg.weight_decay}]
    else:
        params += [{'params': [values], 'lr': cfg.initial_lr, 'weight_decay': 0.0}]

optimizer = optim.Adam(params)

# load pretrained model
model.load(cfg.loadpath, optimizer)

if torch.cuda.is_available():
    model = model.cuda()
    device = torch.device('cuda')
    print("Device: CUDA\n")
else:
    device = torch.device('cpu')
    print("Device: CPU\n")

model.to(device)
model.eval()

# defone dataloader
valid_dataset = FieldsDataset(valid_folders, augment=False, 
                              size=cfg.size, normalize=True)

valid_loader = DataLoader(valid_dataset, batch_size=cfg.BS, 
                          shuffle=False, num_workers=4, 
                          pin_memory=True, drop_last=False)

def evaluate(data_loader):
    """"Perform model evaluation using data_loader (train, valid or test). """

    model.eval()
    dice = DiceLoss()
    bce = nn.BCELoss()
    iou = IoU()

    with torch.no_grad():
        dice_losses = []
        bce_losses = []
        losses = []
        iou_metrics = []

        for batch_idx, data in enumerate(data_loader):
            imgs, masks, _ = data
            imgs, masks = imgs.to(device), masks.to(device)

            pred = model.generate(imgs)
            step = model.get_step()

            dice_loss = dice((pred>0.5).float(), masks)
            bce_loss = bce(pred, masks)
            loss = dice_loss + bce_loss
            iou_value = iou((pred>0.5).float(), masks)

            dice_losses.append(dice_loss.item())
            bce_losses.append(bce_loss.item())
            losses.append(loss.item())
            iou_metrics.append(iou_value.item())

        average_dice_loss = sum(dice_losses) / len(dice_losses)
        average_bce_loss = sum(bce_losses) / len(bce_losses)
        average_loss = sum(losses) / len(losses)
        average_iou = sum(iou_metrics) / len(iou_metrics)

    print("------------------------------------")
    print("Evaluation")
    print('--------')
    print('Losses:')
    print(f'DICE: {average_dice_loss:.4f}, BCE: {average_bce_loss:.4f}')
    print(f'Total Loss: {average_loss:.4f}.')
    print('--------')
    print('Metrics:')
    print(f'IOU: {average_iou:.4f}')
    print('------------------------------------')


if __name__ == "__main__":
    evaluate(valid_loader)
