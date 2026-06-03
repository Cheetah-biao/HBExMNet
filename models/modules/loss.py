import torch
import torch.nn as nn
from models.modules.discriminator_vgg_arch import VGGFeatureExtractor
import torch.nn.functional as F
from models.modules.Subnet_constructor import soft_skel

class ReconstructionLoss(nn.Module):
    def __init__(self, losstype='l2', eps=1e-6, l_mip=0):
        super(ReconstructionLoss, self).__init__()
        self.losstype = losstype
        self.eps = eps
        self.l_mip = l_mip

    def forward(self, x, target):
        if self.l_mip > 0:
            x = torch.cat((self.l_mip * torch.max(x, 0, keepdim=True)[0], x), 0)
            target = torch.cat((self.l_mip * torch.max(target, 0, keepdim=True)[0], target), 0)
        # x_max = x.max()
        # target_max = target.max()
        if x.ndim == 4:
            x = x.unsqueeze(2)
            target = target.unsqueeze(2)
        if self.losstype == 'l2':
            # return torch.mean(torch.sum((x - target) ** 2, (1, 2, 3, 4)))
            return torch.mean(torch.mean((x - target) ** 2))
        elif self.losstype == 'l1':
            diff = x - target
            # loss = torch.mean(torch.sum(torch.sqrt(diff * diff + self.eps), (1, 2, 3, 4)))
            loss = torch.mean(torch.mean(torch.sqrt(diff * diff + self.eps)))

            return loss
        else:
            print("reconstruction loss type error!")
            return 0


class DarkChannelLoss(nn.Module):
    def __init__(self, kernel_size=15, reduction='var', eps=1e-6):
        super().__init__()
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.reduction = reduction
        self.pad = (self.kernel_size[1] // 2, self.kernel_size[1] // 2,
                    self.kernel_size[0] // 2, self.kernel_size[0] // 2)
        self.eps = eps  # 防止数值溢出

    def forward(self, x):
        assert x.dim() == 5 and x.size(1) == 1, "Input must be [B, 1, D, H, W]"

        # 确保输入在 [0, 1] 范围内
        x_normalized = (x - x.min()) / (x.max() - x.min() + self.eps)

        # 独立处理每个 D 切片
        B, C, D, H, W = x_normalized.shape
        x_2d = x_normalized.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)

        # 2D 暗通道计算（安全版本）
        x_pad = F.pad(x_2d + self.eps, self.pad, mode='reflect')
        dark = -F.max_pool2d(-x_pad, kernel_size=self.kernel_size, stride=1) - self.eps
        dark = dark.reshape(B, D, C, H, W).permute(0, 2, 1, 3, 4)
        slice_var = torch.var(x, dim=(3, 4), keepdim=True)  # [B, 1, D, 1, 1]
        loss_var = torch.mean(slice_var)

        return loss_var


# Define GAN loss: [vanilla | lsgan | wgan-gp]
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, x, gt):
        batch_size = x.size()[0]
        h_x = x.size()[-2]
        w_x = x.size()[-1]
        count_h = self._tensor_size(x[:, :, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, :, 1:])
        h_tv = torch.pow((x[:, :, :, 1:, :] - x[:, :, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, :, 1:] - x[:, :, :, :, :w_x - 1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[-2] * t.size()[-1]


class SparseLoss(nn.Module):
    def __init__(self):
        super(SparseLoss, self).__init__()

    def forward(self, x):
        x = torch.abs(x)
        loss = torch.mean(x)
        return loss


class FrequencyLoss(nn.Module):
    def __init__(self):
        super(FrequencyLoss, self).__init__()

    def forward(self, x, target):
        loss_mean = []
        b, c, d, h, w = x.size()
        x = x.contiguous().view(-1, d, h, w)
        target = target.contiguous().view(-1, d, h, w)
        x_fft = torch.fft.fftn(x, dim=(-3, -2, -1))
        x_fft = torch.stack((x_fft.real, x_fft.imag), -1)
        target_fft = torch.fft.fftn(target, dim=(-3, -2, -1))
        target_fft = torch.stack((target_fft.real, target_fft.imag), -1)

        _, d, h, w, f = x_fft.size()

        x_fft = x_fft.view(b, c, d, h, w, f)
        target_fft = target_fft.view(b, c, d, h, w, f)
        diff = x_fft - target_fft
        mask_75 = torch.zeros_like(diff)
        mask_75[:, :, d // 8:7 * d // 8, h // 8:7 * h // 8, w // 8:7 * w // 8, :] = 1
        diff = mask_75 * diff
        loss = torch.mean(torch.mean(diff ** 2, (1, 2, 3, 4, 5)))
        # inner_product = (x_fft * target_fft).sum(dim=-1)
        # norm1 = (x_fft.pow(2).sum(dim=-1)+1e-20).pow(0.5)
        # norm2 = (target_fft.pow(2).sum(dim=-1)+1e-20).pow(0.5)
        # cos = inner_product / (norm1*norm2 + 1e-20)
        # loss_mean.append(-1.0*cos.mean())
        # loss_mean = torch.tensor(loss_mean)
        # loss = torch.mean(loss_mean)
        return loss


class Self_FrequencyLoss(nn.Module):
    def __init__(self):
        super(Self_FrequencyLoss, self).__init__()

    def forward(self, x):
        loss_mean = []
        b, c, d, h, w = x.size()
        x = x.contiguous().view(-1, d, h, w)
        x_fft = torch.fft.fftn(x, dim=(-3, -2, -1))
        x_fft = torch.stack((x_fft.real, x_fft.imag), -1)
        _, d, h, w, f = x_fft.size()

        x_fft = x_fft.view(b, c, d, h, w, f)

        x_fft_half = x_fft[:, :, :, 7 * h // 8:, 7 * w // 8:, :]

        loss = torch.mean(torch.mean(torch.abs(x_fft_half), (1, 2, 3, 4, 5)))
        # inner_product = (x_fft * target_fft).sum(dim=-1)
        # norm1 = (x_fft.pow(2).sum(dim=-1)+1e-20).pow(0.5)
        # norm2 = (target_fft.pow(2).sum(dim=-1)+1e-20).pow(0.5)
        # cos = inner_product / (norm1*norm2 + 1e-20)
        # loss_mean.append(-1.0*cos.mean())
        # loss_mean = torch.tensor(loss_mean)
        # loss = torch.mean(loss_mean)
        return loss


class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 criterion='l1',
                 l_mip=0,
                 ):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm,
            l_mip=l_mip,
        )

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.MSELoss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        return percep_loss


def L1_Charbonnier_loss(X, Y):
    eps = 1e-6
    diff = torch.add(X, -Y)
    error = torch.sqrt(diff * diff + eps)
    loss = torch.sum(error) / torch.numel(error)
    return loss


class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()

    def forward(self, prediction, target):
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(
            0).unsqueeze(0).to(prediction.device)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(
            0).unsqueeze(0).to(prediction.device)

        prediction = F.pad(prediction, (1, 1, 1, 1, 0, 0), 'reflect')
        target = F.pad(target, (1, 1, 1, 1, 0, 0), 'reflect')

        prediction_edges_x = F.conv3d(prediction, sobel_x, padding=0)
        prediction_edges_y = F.conv3d(prediction, sobel_y, padding=0)

        target_edges_x = F.conv3d(target, sobel_x, padding=0)
        target_edges_y = F.conv3d(target, sobel_y, padding=0)

        loss_x = L1_Charbonnier_loss(prediction_edges_x, target_edges_x)
        loss_y = L1_Charbonnier_loss(prediction_edges_y, target_edges_y)

        # edges = torch.sqrt(prediction_edges_x ** 2 + prediction_edges_y ** 2)

        total_loss = loss_x + loss_y

        return total_loss

class SoftSkelLoss(nn.Module):
    def __init__(self, losstype='l1', iter_=10, eps=1e-6):
        super(SoftSkelLoss, self).__init__()
        self.losstype = losstype
        self.eps = eps
        self.iter_ = iter_

    def forward(self, x, target):
        x = soft_skel(x, iter_=self.iter_)
        target = soft_skel(target, iter_=self.iter_)

        if self.losstype == 'l2':
            # return torch.mean(torch.sum((x - target) ** 2, (1, 2, 3, 4)))
            return torch.mean(torch.mean((x - target) ** 2))
        elif self.losstype == 'l1':
            diff = x - target
            # loss = torch.mean(torch.sum(torch.sqrt(diff * diff + self.eps), (1, 2, 3, 4)))
            loss = torch.mean(torch.mean(torch.sqrt(diff * diff + self.eps)))

            return loss
        else:
            print("reconstruction loss type error!")
            return 0
class PearsonCorrelationLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(PearsonCorrelationLoss, self).__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (B, C, D, H, W)
        y: Tensor of shape (B, C, D, H, W)
        """
        B, C, D, H, W = x.shape
        x = x.view(B, C, -1)
        y = y.view(B, C, -1)

        x_mean = x.mean(dim=-1, keepdim=True)
        y_mean = y.mean(dim=-1, keepdim=True)

        x_centered = x - x_mean
        y_centered = y - y_mean

        numerator = (x_centered * y_centered).sum(dim=-1)
        denominator = torch.sqrt((x_centered ** 2).sum(dim=-1) * (y_centered ** 2).sum(dim=-1)) + self.eps

        corr = numerator / denominator  # shape: (B, C)
        loss = 1 - corr  # Pearson loss: 1 - correlation

        return loss.mean()



if __name__ == '__main__':
    import tifffile
    import numpy as np

    x = torch.randn(2, 1, 8, 64, 64)
    y = torch.randn(2, 1, 8, 64, 64)
    loss = PearsonCorrelationLoss()(x, y)
    print(loss)
