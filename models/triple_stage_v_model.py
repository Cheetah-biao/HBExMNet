# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import logging

import tifffile
import torch
import torch.nn.functional as F
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
from models.base_model import BaseModel
import models.networks as networks
from torch.nn.parallel import DataParallel, DistributedDataParallel
from utils.logger import get_logger
from utils.util import save_img, tensor2img
from utils.dist_util import get_dist_info
import time
import os
import re
import imageio
import numpy as np
from scipy.ndimage import zoom

logger = logging.getLogger('base')


def normalize_percentile(im, low, high):
    """Normalize the input 'im' by im = (im - p_low) / (p_high - p_low), where p_low/p_high is the 'low'th/'high'th percentile of the im
    """
    p_low, p_high = np.percentile(im, low), np.percentile(im, high)
    return normalize_min_max(im, max_v=p_high, min_v=p_low)


def normalize_min_max(im, max_v, min_v=0):
    eps = 1e-10
    try:
        import numexpr
        im = numexpr.evaluate("(im - min_v) / (max_v - min_v + eps)")
    except ImportError:
        im = (im - min_v) / (max_v - min_v + eps)
    return im


class Triple_stage_v_Model(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(Triple_stage_v_Model, self).__init__(opt)

        # define network
        self.opt = opt
        if opt['dn_label']:
            self.net_denoise = networks.define_C(opt).to(self.device)
        if opt['sr_label']:
            self.net1 = networks.define_Net1(opt).to(self.device)
            self.netSR = networks.define_Net_SR(opt).to(self.device)
        if opt['dist']:
            if opt['dn_label']:
                self.net_denoise = DistributedDataParallel(self.net_denoise, device_ids=[torch.cuda.current_device()])
            if opt['sr_label']:
                self.net1 = DistributedDataParallel(self.net1, device_ids=[torch.cuda.current_device()])
                self.netSR = DistributedDataParallel(self.netSR, device_ids=[torch.cuda.current_device()])
        else:
            if opt['dn_label']:
                self.net_denoise = DataParallel(self.net_denoise)
            if opt['sr_label']:
                self.net1 = DataParallel(self.net1)
                self.netSR = DataParallel(self.netSR)
        # load pretrained models
        if opt['dn_label']:
            load_path_denoise = self.opt['path']['pretrain_model_C']
            if load_path_denoise is not None:
                logger.info('Loading model for Denoise [{:s}] ...'.format(load_path_denoise))
                self.load_network(load_path_denoise, self.net_denoise, self.opt['path']['strict_load'])
        if opt['sr_label']:
            load_path_net1 = self.opt['path']['pretrain_model_net1']
            load_path_SR = self.opt['path']['pretrain_model_SR']
            if load_path_net1 is not None:
                logger.info('Loading model for Net1 [{:s}] ...'.format(load_path_net1))
                self.load_network(load_path_net1, self.net1, self.opt['path']['strict_load'])
            if load_path_SR is not None:
                logger.info('Loading model for SR [{:s}] ...'.format(load_path_SR))
                self.load_network(load_path_SR, self.netSR, self.opt['path']['strict_load'])


        over_lap = self.opt['val']['over_lap']
        self.low_p = self.opt['val']['low_p']
        self.high_p = self.opt['val']['high_p']
        self.mode = self.opt['val']['mode']
        if opt['sr_label']:
            self.scale = int(opt['scale'])
            crop_size_d = self.opt['val']['crop_size_d']
            crop_size_h = self.opt['val']['crop_size_h']
            crop_size_w = self.opt['val']['crop_size_w']
        else:
            crop_size_d = 1
            crop_size_h = 4
            crop_size_w = 4
            self.scale = 1
        self.factor = self.scale
        self.block_size = (crop_size_d, crop_size_h, crop_size_w)
        self.over_lap = over_lap
        self.dtype = np.float32

    def feed_data(self, data):
        self.lq = data['LQ'].to(self.device)  # LQ

    def __normalize_percentile(self, im, low=0.2, high=99.99):
        # def __normalize_percentile(self, im, low=0.2, high=99.5):
        return normalize_percentile(im.astype(self.dtype), low=low, high=high)

    def __normalize(self, im, max_v):
        im = im.astype(self.dtype)
        return normalize_min_max(im, max_v)

    def __check_dims(self, im, block_size, overlap):
        # To-Do: deal with float block_size and overlap
        def __broadcast(val):
            val = [val for i in im.shape] if np.isscalar(val) else val
            return val

        def __check_block_size(block_size, di_int=8):
            block_shape_new = [int(8 * np.ceil(b / di_int)) for b in block_size]
            return block_shape_new

        assert len(im.shape) == 3 or len(im.shape) == 2, 'Error:the input image must be in shape [depth, height, width]'

        block_size = [r // b + 2 * overlap if b > 1 else r for r, b in zip(im.shape, block_size)]
        block_size = __check_block_size(block_size)
        block_size = __broadcast(block_size)
        overlap = __broadcast(overlap)
        assert len(block_size) == len(
            im.shape), 'Error:ndim of block_size ({}) mismatch that of image size ({})'.format(block_size, im.shape)
        assert len(overlap) == len(im.shape), 'Error:ndim of overlap ({}) mismatch that of image size ({})'.format(
            overlap, im.shape)

        # block_size = [b if b <= i else i for b, i in zip(block_size, im.shape)]

        overlap = [i if i > 1 else i * s for i, s in zip(overlap, block_size)]
        overlap = [0 if b >= s else i for i, b, s in zip(overlap, block_size,
                                                         im.shape)]  # no overlap along the dims where the image size equal to the block size
        overlap = [i if i % 2 == 0 else i + 1 for i in overlap]  # overlap must be even number

        block_size = [b - 2 * i for b, i in zip(block_size, overlap)]  # real block size when inference

        overlap = [int(i) for i in overlap]
        block_size = [int(i) for i in block_size]
        print('block size (overlap excluded) : {} overlap : {}'.format(block_size, overlap))

        return block_size, overlap

    def _padding_block(self, im, blk_size, overlap):
        grid_dim = [int(np.ceil(float(i) / b)) for i, o, b in zip(im.shape, overlap, blk_size)]
        im_size_padded = [(g * b + b if o != 0 else g * b) for g, b, o in zip(grid_dim, blk_size, overlap)]

        # im_wrapped = np.ones(im_size_padded, dtype=self.dtype) * np.min(im)
        # im_wrapped = np.ones(im_size_padded, dtype=self.dtype) * np.percentile(im, 1)

        valid_region = [slice(o // 2, o // 2 + i) for o, i in zip(overlap, im.shape)]
        padding = tuple([(o // 2, k - i - o // 2) for o, i, k in zip(overlap, im.shape, im_size_padded)])

        im_wrapped = np.pad(im, padding, mode='reflect')
        # tifffile.imwrite(r'K:\livingcell\561\test.tif', im_wrapped)

        sr_valid_region = [slice(o // 2 * self.factor, (o // 2 + i) * self.factor) for o, i in zip(overlap, im.shape)]
        print('raw image size : {}, wrapped into : {}'.format(im.shape, im_size_padded))
        print('valid region index: {} ({} after SR)'.format(valid_region, sr_valid_region))

        # im_wrapped[tuple(valid_region)] = im

        return im_wrapped, sr_valid_region

    def __region_iter(self, im, blk_size, overlap, factor):
        """
        Params:
            -im: ndarray in dims of [depth, height, width]
        """
        im_size = im.shape

        anchors = [(z, y, x)
                   for z in range(overlap[0], im_size[0], blk_size[0])
                   for y in range(overlap[1], im_size[1], blk_size[1])
                   for x in range(overlap[2], im_size[2], blk_size[2])]

        for i, anchor in enumerate(anchors):
            # revised_overlap = [0 if a == i else i for a, i in zip(anchor, overlap)]
            begin = [p - c for p, c in zip(anchor, overlap)]
            end = [p + b + c for p, b, c in zip(anchor, blk_size, overlap)]
            yield [slice(b, e) for b, e in zip(begin, end)], \
                [slice((b + c // 2) * factor, (e - c // 2) * factor) for b, e, c, in zip(begin, end, overlap)], \
                [slice((c // 2) * factor, (b + c + c // 2) * factor) for b, c in zip(blk_size, overlap)]

    def __predict_block_denoise(self, block):
        # 动态的多阶段网络使用该配置可以加速10%+
        torch.backends.cudnn.benchmark = True
        block = block[np.newaxis, np.newaxis, :]
        block = torch.from_numpy(block).float()
        block = block.cuda()
        self.net_denoise.eval()
        with torch.no_grad():
            net_out = self.net_denoise(block).squeeze()
            out = net_out.float().cpu().numpy()
            del net_out
            del block
            return out

    def __predict_block_sr(self, block):
        # 动态的多阶段网络使用该配置可以加速10%+
        torch.backends.cudnn.benchmark = True
        b_shape = block.shape
        block = block[np.newaxis, np.newaxis, :]
        block = torch.from_numpy(block).float()
        block = block.cuda()
        self.net1.eval()
        self.netSR.eval()
        with torch.no_grad():
            net1_out = self.net1(block)
            if self.opt['Net_SR']['which_model'] == 'HAT':
                net_out = self.netSR(x=net1_out, size=b_shape).squeeze()
            else:
                net_out = self.netSR(net1_out).squeeze()
            out = net_out.float().cpu().numpy()
            del net_out
            del block
            return out

    def __predict_block_all(self, block):
        torch.backends.cudnn.benchmark = True
        b_shape = block.shape
        block = block[np.newaxis, np.newaxis, :]
        block = torch.from_numpy(block).float()
        block = block.cuda()
        self.net_denoise.eval()
        self.net1.eval()
        self.netSR.eval()
        with torch.no_grad():
            denoise_out = self.net_denoise(block)
            net1_out = self.net1(denoise_out)
            if self.opt['Net_SR']['which_model'] == 'HAT':
                net_out = self.netSR(x=net1_out, size=b_shape).squeeze()
            else:
                net_out = self.netSR(net1_out).squeeze()
            out = net_out.float().cpu().numpy()
            del net_out
            del block
            return out

