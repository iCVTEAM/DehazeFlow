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

import importlib

import torch
import logging
import models.modules.RRDBNet_arch as RRDBNet_arch

logger = logging.getLogger('base')


def find_model_using_name(model_name):
    model_filename = "models.modules." + model_name + "_arch"
    modellib = importlib.import_module(model_filename)

    model = None
    target_model_name = model_name.replace('_Net', '')
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower():
            model = cls

    if model is None:
        print(
            "In %s.py, there should be a subclass of torch.nn.Module with class name that matches %s." % (
                model_filename, target_model_name))
        exit(0)

    return model


####################
# define network
####################

def define_Flow(opt, step):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    Arch = find_model_using_name(which_model)
    netG = Arch(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                nf=opt_net['nf'], nb=opt_net['nb'], scale=opt['scale'], K=opt_net['flow']['K'], opt=opt, step=step)

    return netG
