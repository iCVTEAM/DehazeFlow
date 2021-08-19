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


import glob
import sys
from collections import OrderedDict

from natsort import natsort

import options.options as option
from Measure import Measure, psnr
from imresize import imresize
from models import create_model
from math import log10
import torch
from utils.util import opt_get
import numpy as np
import pandas as pd
import os
import cv2
import random


def fiFindByWildcard(wildcard):
    return natsort.natsorted(glob.glob(wildcard, recursive=True))


def load_model(conf_path):
    opt = option.parse(conf_path, is_train=False)
    opt['gpu_ids'] = '0'
    opt = option.dict_to_nonedict(opt)
    model = create_model(opt)

    model_path = opt_get(opt, ['model_path'], None)
    model.load_network(load_path=model_path, network=model.netG)
    return model, opt


def predict(model, lr):
    model.feed_data({"LQ": t(lr)}, need_GT=False)
    model.test()
    visuals = model.get_current_visuals(need_GT=False)
    return visuals.get('rlt', visuals.get("SR"))


def t(array): return torch.Tensor(np.expand_dims(array.transpose([2, 0, 1]), axis=0).astype(np.float32)) / 255


def rgb(t): return (
        np.clip((t[0] if len(t.shape) == 4 else t).detach().cpu().numpy().transpose([1, 2, 0]), 0, 1) * 255).astype(
    np.uint8)


def imread(path):
    # return cv2.imread(path)[:, :, [2, 1, 0]]
    return cv2.imread(path)


def imwrite(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)


def imCropCenter(img, size):
    h, w, c = img.shape

    h_start = max(h // 2 - size // 2, 0)
    h_end = min(h_start + size, h)

    w_start = max(w // 2 - size // 2, 0)
    w_end = min(w_start + size, w)

    return img[h_start:h_end, w_start:w_end]

def imCropCenter2(img, size):
    h, w, c = img.shape

    h_start = max(h // 2 - size[0] // 2, 0)
    h_end = min(h_start + size[0], h)

    w_start = max(w // 2 - size[1] // 2, 0)
    w_end = min(w_start + size[1], w)

    return img[h_start:h_end, w_start:w_end]


def impad(img, top=0, bottom=0, left=0, right=0, color=255):
    return np.pad(img, [(top, bottom), (left, right), (0, 0)], 'reflect')


def main():
    conf_path = 'DehazeFlow.yml'
    conf = conf_path.split('/')[-1].replace('.yml', '')
    model, opt = load_model(conf_path)
    opt['rank'] = 0
    model.to('cuda')

    hr_dir = opt['dataroot_GT']
    lr_dir = opt['dataroot_HZ']

    if opt['test_mode'] == 'outdoor':
        lr_paths = fiFindByWildcard(os.path.join(lr_dir, '*.jpg'))
    else:
        lr_paths = fiFindByWildcard(os.path.join(lr_dir, '*.png'))
    hr_paths = fiFindByWildcard(os.path.join(hr_dir, '*.png'))

    if opt['test_mode'] == 'indoor':
        temp_list = []
        for i in range(len(hr_paths)):
            for j in range(10):
                temp_list.append(hr_paths[i])
        hr_paths = temp_list


    heat = opt['heat']
    tag = 4
    this_dir = os.path.dirname(os.path.realpath(__file__))
    # test_dir = os.path.join(this_dir, '..', 'results', conf,opt['model_path'].split('/')[-1].replace('.pth', ''),str(heat))
    test_dir = os.path.join(this_dir, '..', 'results', 'test1', str(heat))
    print(f"Out dir: {test_dir}")

    measure = Measure(use_gpu=False)

    fname = f'measure_full_{tag}.csv'
    fname_tmp = fname + "_"
    path_out_measures = os.path.join(test_dir, fname_tmp)
    path_out_measures_final = os.path.join(test_dir, fname)

    if os.path.isfile(path_out_measures_final):
        df = pd.read_csv(path_out_measures_final)
    elif os.path.isfile(path_out_measures):
        df = pd.read_csv(path_out_measures)
    else:
        df = None

    scale = 1

    pad_factor = 8
    AVG_PSNR = 0
    for lr_path, hr_path, idx_test in zip(lr_paths, hr_paths, range(len(lr_paths))):

        if opt['test_mode'] == 'indoor':
            lr = imCropCenter2(imread(lr_path),(456,616))
            hr = imCropCenter2(imread(hr_path),(456,616))
        else:
            lr = imread(lr_path)
            hr = imread(hr_path)

        # Pad image to be % 8
        h, w, c = lr.shape
        lr = impad(lr, bottom=int(np.ceil(h / pad_factor) * pad_factor - h),
                   right=int(np.ceil(w / pad_factor) * pad_factor - w))
        hr = impad(hr, bottom=int(np.ceil(h / pad_factor) * pad_factor - h),
                   right=int(np.ceil(w / pad_factor) * pad_factor - w))

        lr_t = t(lr)
        lr_t = lr_t.cuda()

        if df is not None and len(df[(df['heat'] == heat) & (df['name'] == idx_test)]) == 1:
            continue

        sr_t = model.get_sr(lq=lr_t, heat=heat)
        # sr_t = model(lr_t)

        sr = rgb(torch.clamp(sr_t, 0, 1))

        path_out_sr = os.path.join(test_dir, "{:0.2f}".format(heat).replace('.', ''), "{:06d}_{}.png".format(idx_test,tag))
        imwrite(path_out_sr, sr)

        meas = OrderedDict(conf=conf, heat=heat, name=idx_test)
        meas['PSNR'], meas['SSIM'], meas['LPIPS'] = measure.measure(sr, hr)

        AVG_PSNR += meas['PSNR']

        str_out = format_measurements(meas)
        print(str_out + ' AVG_PSNR: {:0.4f}'.format(AVG_PSNR/(idx_test+1)))

        df = pd.DataFrame([meas]) if df is None else pd.concat([pd.DataFrame([meas]), df])

    df = pd.concat([pd.DataFrame(df.mean()), df],axis=0)
    df.to_csv(path_out_measures, index=False)
    os.rename(path_out_measures, path_out_measures_final)

    str_out = format_measurements(df.mean())
    print(f"Results in: {path_out_measures_final}")
    print(opt['model_path'])
    print('Mean: ' + str_out)


def format_measurements(meas):
    s_out = []
    for k, v in meas.items():
        v = f"{v:0.4f}" if isinstance(v, float) else v
        s_out.append(f"{k}: {v}")
    str_out = ", ".join(s_out)
    return str_out


if __name__ == "__main__":
    main()
