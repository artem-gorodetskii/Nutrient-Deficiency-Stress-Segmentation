# -*- coding: utf-8 -*-
# @Author: Artem Gorodetskii
# @Created Time: 3/22/2022 4:45 PM

import random
import os
import numpy as np
import torch
import torch.nn as nn
import cv2
from matplotlib import pyplot as plt


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def np_now(x: torch.Tensor):
    return x.detach().cpu().numpy()


def print_log(epoch: int, step: int,
              dice_loss: float, bce_loss: float, loss: float, iou: float, 
              start_time: float, end_time: float, average_time: float) -> None:

    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)

    print("------------------------------------")
    print(f"Epoch {epoch}, step {step}.")
    print(f"Time: {int(hours):0>2}:{int(minutes):0>2}:{seconds:05.2f}")
    print('--------')
    print(f'Steps/s: {round(1./average_time, 2)}')
    print('--------')
    print('Losses:')
    print(f'DICE: {dice_loss:.4f}, BCE: {bce_loss:.4f}')
    print(f'Total Loss: {loss:.4f}.')
    print('--------')
    print('Metrics:')
    print(f'IOU: {iou:.4f}')
    print('------------------------------------')


def adjust_learning_rate(current_iter, optimizer, init_lr, gamma, list_of_iters):
    current_lr = 0
    power = 0
    if current_iter < list_of_iters[0]:
        current_lr = init_lr
    elif current_iter > list_of_iters[-1]:
        current_lr = init_lr * (gamma ** len(list_of_iters))
    else:
        list_of_iters.sort(reverse=True)
        nearest_smaller_iter = min(list_of_iters, key=lambda x : x - current_iter > 0 )
        list_of_iters.sort(reverse=False)
        index_of_nearest_smaller_iter = list_of_iters.index(nearest_smaller_iter) 
        power = index_of_nearest_smaller_iter + 1
        current_lr = init_lr * (gamma ** power)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr

    return current_lr


class DiceLoss(nn.Module):
    def __init__(self, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class IoU(nn.Module):
    def __init__(self, size_average=True):
        super(IoU, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        iou = (intersection + smooth) / (union + smooth)

        return iou

class ValueWindow:
    def __init__(self, window_size=100):
        self._window_size = window_size
        self._values = []

    def append(self, x):
        self._values = self._values[-(self._window_size - 1):] + [x]

    @property
    def sum(self):
        return sum(self._values)

    @property
    def count(self):
        return len(self._values)

    @property
    def average(self):
        return self.sum / max(1, self.count)

    def reset(self):
        self._values = []


def show_images(folderpath, pred_mask=None):
    "Plot input images, target mask and predicted mask. "

    image_0 = cv2.imread(os.path.join(folderpath, 'image_i0.png'))
    image_1 = cv2.imread(os.path.join(folderpath, 'image_i1.png'))
    image_2 = cv2.imread(os.path.join(folderpath, 'image_i2.png'))

    image_0 = cv2.cvtColor(image_0, cv2.COLOR_BGR2RGB)/ 255.0
    image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)/ 255.0
    image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)/ 255.0

    target_mask = cv2.imread(os.path.join(folderpath, 'nutrient_mask_g0.png'), 0)/ 255.0

    plt.figure(figsize=(25, 25))

    plt.subplot(1, 5, 1)
    plt.imshow(image_0)
    plt.axis("off")
    plt.title("image 0")

    plt.subplot(1, 5, 2)
    plt.imshow(image_1)
    plt.axis("off")
    plt.title("image 1")

    plt.subplot(1, 5, 3)
    plt.imshow(image_2)
    plt.axis("off")
    plt.title("image 2")

    plt.subplot(1, 5, 4)
    plt.imshow(target_mask, 'gray')
    plt.axis("off")
    plt.title("target mask")

    plt.subplot(1, 5, 5)
    plt.imshow(pred_mask, 'gray')
    plt.axis("off")
    plt.title("predicted mask")

    plt.show()