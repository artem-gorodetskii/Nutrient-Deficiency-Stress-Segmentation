# -*- coding: utf-8 -*-
# @Author: Artem Gorodetskii
# @Created Time: 3/22/2022 4:45 PM

import torch
from skimage import transform
import numpy as np
import cv2
import os


def inference_one_sample(model, folderpath, cfg, device, threshold = 0.5):
    """Make predictions for single data example. """

    image_0 = cv2.imread(os.path.join(folderpath, 'image_i0.png'))
    image_1 = cv2.imread(os.path.join(folderpath, 'image_i1.png'))
    image_2 = cv2.imread(os.path.join(folderpath, 'image_i2.png'))

    image_0 = cv2.cvtColor(image_0, cv2.COLOR_BGR2RGB)/ 255.0
    image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)/ 255.0
    image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)/ 255.0

    original_size = image_0.shape[0], image_0.shape[1]

    image_0 = transform.resize(image=image_0, output_shape=(cfg.size, cfg.size), order=1)
    image_1 = transform.resize(image=image_1, output_shape=(cfg.size, cfg.size), order=1)
    image_2 = transform.resize(image=image_2, output_shape=(cfg.size, cfg.size), order=1)

    image = np.zeros(((cfg.size, cfg.size, 9)))
    image[:, :, :3] = image_0
    image[:, :, 3:6] = image_1
    image[:, :, 6:9] = image_2

    image[:, :, :] = (image[:, :, :] - cfg.channels_avgs[:]) / cfg.channels_stds[:]

    image = torch.as_tensor(image).float().contiguous()
    image = image.permute(2, 0, 1).unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        pred = model.generate(image)

    pred = pred.squeeze(0).squeeze(0)
    pred = pred.detach().cpu().numpy()
    pred = transform.resize(image=pred, output_shape=(original_size[0], original_size[1]), order=1)

    pred = (pred>=threshold).astype('int8')

    return pred