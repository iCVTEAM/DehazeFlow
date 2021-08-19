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
# This file contains content licensed by https://github.com/chaiyujin/glow-pytorch/blob/master/LICENSE

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

from models.modules import thops


class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels, LU_decomposed=False):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        self.w_shape = w_shape
        self.LU = LU_decomposed

    def get_weight(self, input, reverse):
        w_shape = self.w_shape
        pixels = thops.pixels(input)
        dlogdet = torch.slogdet(self.weight)[1] * pixels
        if not reverse:
            weight = self.weight.view(w_shape[0], w_shape[1], 1, 1)
        else:
            weight = torch.inverse(self.weight.double()).float() \
                .view(w_shape[0], w_shape[1], 1, 1)
        return weight, dlogdet
    def forward(self, input, logdet=None, reverse=False):
        """
        log-det = log|abs(|W|)| * pixels
        """
        weight, dlogdet = self.get_weight(input, reverse)
        if not reverse:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet + dlogdet
            return z, logdet
        else:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet - dlogdet
            return z, logdet
# import argparse
# import time
# import options.options as option
# from torch.nn.parallel import DataParallel, DistributedDataParallel
# import torch.distributed as dist
# parser = argparse.ArgumentParser()
# parser.add_argument('-opt', type=str, help='Path to option YMAL file.')
# parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
#                     help='job launcher')
# parser.add_argument('--local_rank', type=int, default=0)
# args = parser.parse_args()
# opt = option.parse(args.opt, is_train=True)
# opt_net = opt['network_G']
# a = InvertibleConv1x1(128,False)
#
# # a = FlowUpsamplerNet((128, 128, 3), 64, 16,
# #                  flow_coupling=opt['network_G']['flow']['coupling'], opt=opt)
#
# a = DataParallel(a,device_ids=[0,1,2,3]).to('cuda')
#
#
# d = torch.randn(4,128,256,256).cuda()
# citer = torch.nn.MSELoss(reduction='mean')
# optimizer_G = torch.optim.Adam(
#     [
#         {"params": a.parameters(), "lr": 0.000001, 'beta1': 0.9,
#          'beta2': 0.99, 'weight_decay': 0}
#     ],
# )
#
# while True:
#     optimizer_G.zero_grad()
#     b = torch.randn(4, 128, 256, 256).cuda()
#     c = torch.randn(4,1).cuda()
#     r1,r2 = a(b,c)
#     time.sleep(1)
#     loss=torch.mean(r2)
#     loss.backward()
#     optimizer_G.step()
#
#     print('Done!')
