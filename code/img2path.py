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
from natsort import natsort
import numpy as np
import os


def fiFindByWildcard(wildcard):
    return natsort.natsorted(glob.glob(wildcard, recursive=True))


def its():

    lr_dir = 'D:/SRFlow-master/datasets/RESIDE/hazy'
    hr_dir = 'D:/SRFlow-master/datasets/RESIDE/trans'

    lr_paths = fiFindByWildcard(os.path.join(lr_dir, '*.png'))

    indices = np.arange(13990)
    np.random.shuffle(indices)

    lr_paths = [lr_paths[i] for i in indices[:13990]]

    with open("../datasets/its_hazy_train.txt", "w") as f:
        for i in range(13990):
            f.write(os.path.join(lr_dir,lr_paths[i])+'\n')
    with open("../datasets/its_clear_train.txt", "w") as f:
        for i in range(13990):
            f.write(os.path.join(hr_dir,lr_paths[i].split('/')[-1].split('_')[0]+'.png')+'\n')

    # for i in range(2):
    #     show_from_np(cv2.imread(os.path.join(lr_dir,lr_paths[i])))
    #     show_from_np(cv2.imread(os.path.join(hr_dir,lr_paths[i].split('/')[-1].split('_')[0]+'.png')))

def ots():
    lr_dir = 'D:\BaiduNetdiskDownload\OTS\hazy'
    hr_dir = 'D:\BaiduNetdiskDownload\OTS\gt'


    lr_paths = fiFindByWildcard(os.path.join(lr_dir, '*.png'))

    indices = np.arange(313950)
    np.random.shuffle(indices)

    lr_paths = [lr_paths[i] for i in indices[:313950]]

    with open("../datasets/its_hazy_val.txt", "w") as f:
        for i in range(len(lr_paths)):
            f.write(os.path.join(lr_dir,lr_paths[i])+'\n')
    with open("../datasets/its_clear_val.txt", "w") as f:
        for i in range(len(lr_paths)):
            f.write(os.path.join(hr_dir,lr_paths[i].split('/')[-1].split('_')[0]+'.png')+'\n')



if __name__ == "__main__":
    its()
    ots()
