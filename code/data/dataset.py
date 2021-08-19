# Copyright (c) 2020 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file contains content licensed by https://github.com/xinntao/BasicSR/blob/master/LICENSE/LICENSE

import os
import subprocess
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
import time
import torch
import h5py
import math
import random
import matplotlib.pyplot as plt
import random
from test import fiFindByWildcard
import cv2

# import pickle

def show_from_np(img, title=None):
    plt.figure()
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def impad(img, top=0, bottom=0, left=0, right=0, color=255):
    return np.pad(img, [(top, bottom), (left, right), (0, 0)], 'reflect')


class PathDataset(data.Dataset):
    def __init__(self, opt, path):
        super(PathDataset, self).__init__()
        self.opt = opt
        self.crop_size = opt.get("GT_size", None)
        self.scale = None
        self.random_scale_list = [1]
        self.path = path
        self.downsize = [0.5,0.7,1.0]

        gpu = True
        augment = True

        self.from_path_list = opt['from_paths_list']
        self.crop = opt['crop'] if "crop" in opt.keys() else True

        self.GT = open(self.path[0],'r').readlines()[:opt['path_num']]
        self.HZ = open(self.path[1],'r').readlines()[:opt['path_num']]


        self.gpu = gpu
        self.augment = augment

        self.measures = None

    def __len__(self):
        return len(self.GT)

    def __getitem__(self, item):
        if self.from_path_list:
            GT = cv2.imread(self.GT[item].strip())
            HZ = cv2.imread(self.HZ[item].strip())

            GT_patch = None
            HZ_patch = None
            if self.crop:
                # downscale = self.downsize[random.randint(0, 2)]
                # GT = downsample(GT, downscale)
                # HZ = downsample(HZ, downscale)
                # GT_patch, HZ_patch = random_crop(GT, HZ, 152)
                GT_patch, HZ_patch = self.crop_img(GT,HZ)
            else:
                #For OTS Validation
                pad_factor = 8
                h, w, c = HZ.shape
                GT_patch = impad(GT, bottom=int(np.ceil(h / pad_factor) * pad_factor - h),
                           right=int(np.ceil(w / pad_factor) * pad_factor - w))
                HZ_patch = impad(HZ, bottom=int(np.ceil(h / pad_factor) * pad_factor - h),
                           right=int(np.ceil(w / pad_factor) * pad_factor - w))

            GT_patch = GT_patch.transpose([2, 0, 1]).astype(np.float32) / 255
            HZ_patch = HZ_patch.transpose([2, 0, 1]).astype(np.float32) / 255

            HZ_patch = np.clip(HZ_patch, 0, 1)  # we might get out of bounds due to noise
            GT_patch = np.clip(GT_patch, 0, 1)  # we might get out of bounds due to noise
            HZ_patch = np.asarray(HZ_patch, np.float32)
            GT_patch = np.asarray(GT_patch, np.float32)

            if self.crop:
                flip_channel = random.randint(0, 1)
                if flip_channel != 0:
                    HZ_patch = np.flip(HZ_patch, 2)
                    GT_patch = np.flip(GT_patch, 2)
                # randomly rotation
                rotation_degree = random.randint(0, 3)
                HZ_patch = np.rot90(HZ_patch, rotation_degree, (1, 2))
                GT_patch = np.rot90(GT_patch, rotation_degree, (1, 2))

            lr = torch.Tensor(HZ_patch.copy())
            hr = torch.Tensor(GT_patch.copy())

            return {'LQ': lr, 'GT': hr, 'LQ_path': str(item), 'GT_path': str(item)}
        else:
            HZ_patch = self.HZ[item]
            GT_patch = self.GT[item]


            lr = torch.Tensor(HZ_patch.copy())
            hr = torch.Tensor(GT_patch.copy())

        return {'LQ': lr, 'GT': hr, 'LQ_path': str(item), 'GT_path': str(item)}

    def crop_img(self, GT, HZ):
        if math.floor(self.downsize[0] * GT.shape[0]) > self.crop_size and math.floor(self.downsize[0] * GT.shape[1]) > self.crop_size:
            downscale = self.downsize[random.randint(0, 2)]
            GT = downsample(GT, downscale)
            HZ = downsample(HZ, downscale)
            return  random_crop(GT, HZ, 152)
        elif math.floor(self.downsize[1] * GT.shape[0]) > self.crop_size and math.floor(self.downsize[1] * GT.shape[1]) > self.crop_size:
            downscale = self.downsize[random.randint(1, 2)]
            GT = downsample(GT, downscale)
            HZ = downsample(HZ, downscale)
            return  random_crop(GT, HZ, 152)
        else:
            return random_crop(GT, HZ, 152)


def downsample(img,scale):
    size_x = math.floor(scale * img.shape[1])
    size_y = math.floor(scale * img.shape[0])
    img = cv2.resize(img, (size_x, size_y), interpolation=cv2.INTER_CUBIC)
    return img


def random_crop(hr, lr, size):

    size_x = lr.shape[0]
    size_y = lr.shape[1]

    start_x = np.random.randint(low=0, high=(size_x - size) + 1) if size_x > size else 0
    start_y = np.random.randint(low=0, high=(size_y - size) + 1) if size_y > size else 0

    # LR Patch
    lr_patch = lr[start_x:start_x + size, start_y:start_y + size,:]

    # HR Patch
    hr_patch = hr[start_x:start_x + size, start_y:start_y + size,:]

    return hr_patch, lr_patch


