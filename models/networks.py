import logging
import models.modules.discriminator_vgg_arch as SRGAN_arch
from models.modules.HAT import HAT
from models.modules.NAFNet import NAFNet
from models.modules.Unet import Unet


logger = logging.getLogger('base')

####################
# define network
####################
def define_C(opt):
    opt_net = opt['network_C']
    which_model = opt_net['which_model_C']
    if which_model == 'Unet':
        netC = Unet(num_features=opt_net['n_fea'])
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netC


# Discriminator
def define_D(opt):
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator_vgg_128':
        netD_HR = SRGAN_arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    elif which_model == 'Unet_discriminator_sn':
        netD_HR = SRGAN_arch.UNetDiscriminatorSN(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    elif which_model == 'NLayerDiscriminator':
        netD_HR = SRGAN_arch.NLayerDiscriminator(in_nc=opt_net['in_nc'], nf=opt_net['nf'], n_layers=opt_net['n_layers'],
                                                 kw=opt_net['kw'])
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netD_HR

def define_Net1(opt):
    scale = 1
    opt_net = opt['Net1']
    which_model = opt_net['which_model']
    if which_model == 'NAFNet':
        Net1 = NAFNet(scale=scale, num_features=opt_net['n_fea'])
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return Net1


def define_Net_SR(opt):
    # scale = 2
    scale = opt['scale']
    opt_net = opt['Net_SR']
    which_model = opt_net['which_model']
    if which_model == 'NAFNet':
        Net_SR = NAFNet(scale=scale, num_features=opt_net['n_fea'])
    elif which_model == 'HAT':
        try:
            img_size_z = 8
            img_size_xy = 32
        except:
            img_size_xy = opt['datasets']['train']['GT_size_xy'] // scale
            img_size_z = opt['datasets']['train']['GT_size_z'] // scale
        Net_SR = HAT(img_size=(img_size_z, img_size_xy, img_size_xy), upscale=scale, depths=opt_net['depth'],
                   num_heads=(opt_net['num_heads']), window_size=opt_net['window_size'],
                   mlp_ratio=opt_net['mlp_ratio'], embed_dim=opt_net['n_fea'])
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return Net_SR
