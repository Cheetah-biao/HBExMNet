import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.module_util import LayerNorm3d
from models.modules.Subnet_constructor import PixelShuffle3d


def default_conv(channel_in, channel_out, kernel_size):
    return nn.Conv3d(channel_in, channel_out, kernel_size, padding=(kernel_size // 2))


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv3d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv3d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv3d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv3d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv3d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm3d(c)
        self.norm2 = LayerNorm3d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFNet(nn.Module):

    def __init__(self, conv=default_conv, scale=4, num_features=48, num_rg=16):
        super().__init__()

        self.scale = scale
        n_feats = num_features
        kernel_size = 3
        act = nn.ReLU(True)

        # define head module
        modules_head = [conv(1, n_feats, kernel_size)]

        # define body module
        modules_body = [
            NAFBlock(c=num_features, drop_out_rate=0) for _ in range(num_rg)
        ]
        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define PixelShuffle module
        # define PixelShuffle module
        modules_up_x2 = [conv(n_feats, 8 * n_feats, 3),
                         PixelShuffle3d(2),
                         ]
        modules_up_x4 = [conv(n_feats, 8 * n_feats, 3),
                         PixelShuffle3d(2),
                         conv(n_feats, 8 * n_feats, 3),
                         PixelShuffle3d(2),
                         ]
        modules_up_x3 = [
                         conv(n_feats, 8 * 27, 3),
                         PixelShuffle3d(3),
                         ]

        # define tail module
        modules_tail_X1 = [conv(n_feats, 1, 3)]
        modules_tail_X2 = [conv(n_feats, 1, 3)]
        modules_tail_X3 = [conv(8, 1, 3)]
        modules_tail_X4 = [conv(n_feats, 1, 3)]


        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)

        self.modules_up_x2 = nn.Sequential(*modules_up_x2)
        self.modules_up_x3 = nn.Sequential(*modules_up_x3)
        self.modules_up_x4 = nn.Sequential(*modules_up_x4)

        self.tail_X1 = nn.Sequential(*modules_tail_X1)
        self.tail_X2 = nn.Sequential(*modules_tail_X2)
        self.tail_X3 = nn.Sequential(*modules_tail_X3)
        self.tail_X4 = nn.Sequential(*modules_tail_X4)


    def forward(self, x):
        res = self.head(x)
        feats = self.body(res)
        feats = res + feats
        if self.scale == 1:
            x = self.tail_X1(feats)
        elif self.scale == 2:
            feats = self.modules_up_x2(feats)
            x = self.tail_X2(feats)
        elif self.scale == 4:
            feats = self.modules_up_x4(feats)
            x = self.tail_X4(feats)
        elif self.scale == 3:
            feats = self.modules_up_x3(feats)
            x = self.tail_X3(feats)
        return x


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
    net = NAFNet(scale=3)
    model_info(net)

    x = torch.randn(2, 1, 8, 32, 32)
    x = net(x)
    print(x.shape)
