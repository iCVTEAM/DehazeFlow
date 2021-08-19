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

'''create dataset and dataloader'''
import logging
import torch
import torch.utils.data


def create_dataloader(dataset, dataset_opt, opt=None, sampler=None):
    phase = dataset_opt.get('phase', 'test')
    if phase == 'train':
        gpu_ids = opt.get('gpu_ids', None)
        gpu_ids = gpu_ids if gpu_ids else []
        num_workers = dataset_opt['n_workers'] * len(gpu_ids)
        batch_size = dataset_opt['batch_size']
        shuffle = False
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size ,sampler=sampler,
                                        drop_last=True,pin_memory=True)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                           pin_memory=True)


def create_dataset(dataset_opt,path):
    # print(dataset_opt)
    mode = dataset_opt['mode']
    if mode == 'PATH':
        from data.dataset import PathDataset as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
    dataset = D(dataset_opt,path)

    return dataset
