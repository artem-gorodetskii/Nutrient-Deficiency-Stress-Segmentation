# -*- coding: utf-8 -*-
# @Author: Artem Gorodetskii
# @Created Time: 3/22/2022 4:45 PM

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import albumentations as A
from skimage import transform
import numpy as np
import cv2
import os

class FieldsDataset(Dataset):
    "Dataset class for the Nutrient Deficiency Stress Segmentation task. "

    def __init__(self, 
                 folderspath: list, 
                 size = 512,
                 channels_avgs = None,
                 channels_stds = None,
                 augment=None,
                 normalize=False):
      
        self.size = size

        # used for input normalization
        self.channels_avgs = channels_avgs
        self.channels_stds = channels_stds

        self.folderspath = folderspath

        self.augment = augment
        self.normalize = normalize

    def __len__(self):
        return len(self.folderspath)

    def __getitem__(self, idx):

        folder = self.folderspath[idx]

        # load all 3 images
        image_0 = cv2.imread(os.path.join(folder, 'image_i0.png'))
        image_1 = cv2.imread(os.path.join(folder, 'image_i1.png'))
        image_2 = cv2.imread(os.path.join(folder, 'image_i2.png'))

        image_0 = cv2.cvtColor(image_0, cv2.COLOR_BGR2RGB)/ 255.0
        image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)/ 255.0
        image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)/ 255.0

        # load masks
        boundary_mask = cv2.imread(os.path.join(folder, 'bounday_mask.png'), 0)/ 255.0
        target_mask = cv2.imread(os.path.join(folder, 'nutrient_mask_g0.png'), 0)/ 255.0

        # resize images
        image_0 = transform.resize(image=image_0, output_shape=(self.size, self.size), order=1)
        image_1 = transform.resize(image=image_1, output_shape=(self.size, self.size), order=1)
        image_2 = transform.resize(image=image_2, output_shape=(self.size, self.size), order=1)
        boundary_mask = transform.resize(image=boundary_mask, output_shape=(self.size, self.size), order=1)
        target_mask  = transform.resize(image=target_mask , output_shape=(self.size, self.size), order=1)

        # combine images
        comb_image = np.zeros(((self.size, self.size, 10)))
        comb_image[:, :, :3] = image_0
        comb_image[:, :, 3:6] = image_1
        comb_image[:, :, 6:9] = image_2
        comb_image[:, :, 9] = boundary_mask

        # input normalization
        if self.normalize and self.channels_avgs is not None and self.channels_stds is not None:
            comb_image[:, :, :9] = (comb_image[:, :, :9] - self.channels_avgs[:]) / self.channels_stds[:]

        # augmentation
        if self.augment:
            aug = A.Compose([A.VerticalFlip(p=0.5),
                             A.HorizontalFlip(p=0.5),
                             A.Rotate(limit=30, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_REFLECT_101, p=1)
                           ])
            augmented = aug(image=comb_image, mask=target_mask)
            comb_image = augmented['image']
            target_mask = augmented['mask']

        comb_image[:, :, :9] = comb_image[:, :, :9]*np.expand_dims(comb_image[:, :, -1], -1)

        image = torch.as_tensor(comb_image[:, :, :9]).float().contiguous()
        boundary_mask = torch.as_tensor(comb_image[:, :, 9]).float().contiguous()
        target_mask = torch.as_tensor(target_mask).float().contiguous()

        image = image.permute(2, 0, 1)

        return image, target_mask.unsqueeze(0), boundary_mask.unsqueeze(0)