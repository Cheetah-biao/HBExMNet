import math
import numpy as np
from math import exp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# import codes.models.modules.module_util as mutil
import models.modules.module_util as mutil


def default_conv(channel_in, channel_out, kernel_size):
    return nn.Conv3d(channel_in, channel_out, kernel_size, padding=(kernel_size // 2))


class BasicBlock(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size):
        super(BasicBlock, self).__init__()
        self.inConv = nn.Sequential(
            nn.Conv3d(channel_in, channel_out, kernel_size, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.inConv(x)


class ResBlock(nn.Module):
    def __init__(
            self, n_feats=1, kernel_size=3, bias=True,
            bn=False, act=(nn.ReLU(True)), res_scale=0.1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(nn.Conv3d(in_channels=n_feats,
                               out_channels=n_feats,
                               kernel_size=kernel_size,
                               padding=1,
                               bias=bias))
            if bn:
                m.append(nn.BatchNorm3d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, withConvReLU=True):
        super(DownSampleBlock, self).__init__()
        if withConvReLU:
            self.conv = nn.Sequential(
                nn.Conv3d(in_channels=in_channels,
                          out_channels=1,
                          kernel_size=3,
                          padding=1,
                          stride=2),
                nn.Conv3d(in_channels=1,
                          out_channels=1,
                          kernel_size=3,
                          padding=1),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv = nn.Conv3d(in_channels=in_channels,
                                  out_channels=1,
                                  kernel_size=3,
                                  padding=1,
                                  stride=2)

    def forward(self, x):
        return self.conv(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      padding=1,
                      stride=1),
            nn.Conv3d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class PixelShuffle3d(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        batch_size, channels, in_depth, in_height, in_width = x.size()
        nOut = channels // self.scale ** 3

        out_depth = in_depth * self.scale
        out_height = in_height * self.scale
        out_width = in_width * self.scale

        input_view = x.contiguous().view(batch_size, nOut, self.scale, self.scale, self.scale, in_depth, in_height,
                                         in_width)

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)



class PixelUnShuffle3d(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        batch_size, n_in, in_depth, in_height, in_width = x.size()

        n_out = n_in * self.scale ** 3

        out_depth = in_depth // self.scale
        out_height = in_height // self.scale
        out_width = in_width // self.scale

        input_view = x.contiguous().view(batch_size, n_in, out_depth, self.scale, out_height, self.scale, out_width,
                                         self.scale)
        output = input_view.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        return output.view(batch_size, n_out, out_depth, out_height, out_width)


class SubVoxelUpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale=2, ):
        super(SubVoxelUpsampleBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels,
                      out_channels=8 * in_channels,
                      kernel_size=3,
                      padding=1),
            PixelShuffle3d(scale),
            nn.Conv3d(in_channels=in_channels,
                      out_channels=1,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv3d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv3d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv3d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv3d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv3d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        mutil.initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5


class DenseBlockp(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(DenseBlockp, self).__init__()
        # self.conv1 = nn.Conv3d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv1 = nn.Conv3d(channel_in, gc, (3, 7, 7), 1, (1, 3, 3), bias=bias)
        self.conv2 = nn.Conv3d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv3d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv3d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv3d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        # self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.act = nn.GELU()

        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        mutil.initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.act(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.act(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5


class DepthWiseConv(nn.Module):
    def __init__(self, channel_in, channel_out, k_size, stride, pad, bias):
        super(DepthWiseConv, self).__init__()
        self.depth_conv = nn.Conv3d(in_channels=channel_in, out_channels=channel_in, kernel_size=k_size, stride=stride,
                                    padding=pad, groups=channel_in, bias=bias)
        self.point_conv = nn.Conv3d(in_channels=channel_in, out_channels=channel_out, kernel_size=1, stride=1,
                                    padding=0, groups=1, bias=bias)

    def forward(self, x):
        out = self.depth_conv(x)
        out = self.point_conv(out)
        return out


class DenseBlockpp(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(DenseBlockpp, self).__init__()
        self.conv1 = nn.Conv3d(channel_in, gc, (3, 7, 7), 1, (1, 3, 3), bias=bias)
        self.conv2 = nn.Conv3d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv3d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv3d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv3d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        self.act = nn.GELU()

        if init == 'xavier':
            mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        mutil.initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.act(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5


def subnet(net_structure, init='xavier'):
    def constructor(channel_in, channel_out, depth):
        if net_structure == 'DBNet':
            if init == 'xavier':
                return DenseBlock(channel_in, channel_out, init)
            else:
                return DenseBlock(channel_in, channel_out)
        elif net_structure == 'DBpNet':
            if init == 'xavier':
                return DenseBlockp(channel_in, channel_out, init)
            else:
                return DenseBlockp(channel_in, channel_out)

        elif net_structure == 'DBppNet':
            if init == 'xavier':
                return DenseBlockpp(channel_in, channel_out, init)
            else:
                return DenseBlockpp(channel_in, channel_out)

        else:
            return None

    return constructor


def ADDnoise(x, gauss_sigma=0.015, poisson_sigma=2800):
    def normalize_min_max(im):
        max_ = torch.max(im)
        min_ = torch.min(im)
        eps = 1e-10
        im = (im - min_) / (max_ - min_ + eps)
        return im

    def add_gauss_noise_torch(img, g_s):
        B, C, D, H, W = img.shape
        noise = torch.normal(0, g_s, size=(B, C, D, H, W)).to(
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        # noise = noise.repeat(1, 1, D, 1, 1)
        img = torch.add(img, noise)
        img = torch.clamp(img, 0, 1)
        return img

    def add_poisson_torch(img, p_s):
        img = torch.multiply(img, p_s)
        img = torch.poisson(img)
        return img

    if gauss_sigma > 0:
        x = add_gauss_noise_torch(x, gauss_sigma)
    if poisson_sigma > 0:
        x = add_poisson_torch(x, poisson_sigma)
    return normalize_min_max(x)


def GaussianBlur(x, kernel_size=3, sigma_x=1., sigma_y=1., sigma_z=1.3):
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_kernel(kernel_size=21., sigma_x=1., sigma_y=1., sigma_z=1.):
        x_1D_window = gaussian(kernel_size, sigma_x).unsqueeze(1)
        y_1D_window = gaussian(kernel_size, sigma_y).unsqueeze(1)
        z_1D_window = gaussian(kernel_size, sigma_z).unsqueeze(1)
        xy_2D_window = x_1D_window.mm(y_1D_window.t())
        _3D_window = z_1D_window.mm(xy_2D_window.reshape(1, -1)).reshape(kernel_size, kernel_size,
                                                                         kernel_size).float().unsqueeze(0).unsqueeze(0)
        kernel = _3D_window.expand(1, 1, kernel_size, kernel_size, kernel_size).contiguous().to(
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        kernel.requires_grad = False
        return kernel

    kernel = create_kernel(kernel_size, sigma_x, sigma_y, sigma_z)
    x = F.conv3d(x, kernel, padding=kernel_size // 2)
    return x

def soft_erode(img):
    if len(img.shape) == 4:
        p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
        return torch.min(p1, p2)
    elif len(img.shape) == 5:
        p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
        p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
        p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
        return torch.min(torch.min(p1, p2), p3)


def soft_dilate(img):
    if len(img.shape) == 4:
        return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))
    elif len(img.shape) == 5:
        return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))


def soft_open(img):
    return soft_dilate(soft_erode(img))


def soft_skel(img, iter_):
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for j in range(iter_):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel

def model_info(model):  # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print('\n%5s %50s %9s %12s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print('%5g %50s %9s %12g %20s %12.3g %12.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g))


if __name__ == '__main__':
    # torch.cuda.empty_cache()
    net = DenseBlockpp(channel_in=32, channel_out=32)
    model_info(net)

    x = torch.randn(2, 64, 8, 16, 128)
    x = net(x)
    print(x.shape)
