import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import ReconstructionLoss, GANLoss, TVLoss, SoftSkelLoss, SparseLoss, PerceptualLoss, \
    FrequencyLoss
from models.modules.Quantization import Quantization

logger = logging.getLogger('base')


class GanModel(BaseModel):
    def __init__(self, opt):
        super(GanModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']
        test_opt = opt['test']
        self.train_opt = train_opt
        self.test_opt = test_opt

        self.out_channel = opt['network_C']['out_nc']

        self.netC = networks.define_C(opt).to(self.device)
        if opt['dist']:
            self.netC = DistributedDataParallel(self.netC, device_ids=[torch.cuda.current_device()])
        else:
            self.netC = DataParallel(self.netC)
        # print network
        self.print_network()
        self.load()

        self.Quantization = Quantization()

        if self.is_train:
            self.netD = networks.define_D(opt).to(self.device)
            if opt['dist']:
                self.netD = DistributedDataParallel(self.netD, device_ids=[torch.cuda.current_device()])
            else:
                self.netD = DataParallel(self.netD)
            self.netC.train()
            self.netD.train()

            self.load_D()

            # loss
            self.Reconstruction_back = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_back'])
            self.TVLoss = TVLoss()
            self.SoftSkelLoss = SoftSkelLoss()
            self.SparseLoss = SparseLoss()
            self.FrequencyLoss = FrequencyLoss()

            # # feature loss
            # if train_opt['lambda_feature'] > 0:
            #     self.Reconstructionf = ReconstructionLoss(losstype=self.train_opt['feature_criterion'])
            #
            #     self.l_fea_w = train_opt['lambda_feature']
            #     self.netF = networks.define_F(opt, use_bn=False).to(self.device)
            #     if opt['dist']:
            #         self.netF = DistributedDataParallel(self.netF, device_ids=[torch.cuda.current_device()])
            #     else:
            #         self.netF = DataParallel(self.netF)
            # else:
            #     self.l_fea_w = 0
            # feature loss
            opt_feature = train_opt['perceptual_opt']
            if train_opt['lambda_feature'] > 0:
                self.Featureloss = PerceptualLoss(
                    layer_weights=opt_feature['layer_weights'],
                    vgg_type='vgg19',
                    use_input_norm=True,
                    range_norm=False,
                    perceptual_weight=1.0,
                    criterion='l1').to(self.device)

                self.l_fea_w = train_opt['lambda_feature']
            else:
                self.l_fea_w = 0

            # GD gan loss
            self.cri_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
            self.l_gan_w = train_opt['lambda_gan']

            # D_update_ratio and D_init_iters
            self.D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] else 1
            self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0

            # optimizers
            # C
            wd_C = train_opt['weight_decay_C'] if train_opt['weight_decay_C'] else 0
            self.optimizer_C = torch.optim.Adam(self.netC.parameters(), lr=train_opt['lr_C'], weight_decay=wd_C,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_C)

            # D
            wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=train_opt['lr_D'], weight_decay=wd_D,
                                                betas=(train_opt['beta1_D'], train_opt['beta2_D']))
            self.optimizers.append(self.optimizer_D)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

    def feed_data(self, data):
        self.real_L = data['LQ'].to(self.device)  # LQ
        self.real_H = data['GT'].to(self.device)  # GT

    def gaussian_batch(self, dims):
        return torch.randn(tuple(dims)).to(self.device)

    def feature_loss(self, fake, real):
        l_g_fea = self.l_fea_w * self.Featureloss(fake, real)

        return l_g_fea

    def loss_backward(self, pred, gt):
        # mse loss
        x_samples_image = pred[:, :self.out_channel, :, :]
        l_back_rec = self.train_opt['lambda_rec_back'] * self.Reconstruction_back(gt, x_samples_image)

        # feature loss
        if self.l_fea_w > 0:
            l_back_fea = self.feature_loss(x_samples_image, gt)
        else:
            l_back_fea = torch.tensor(0)

        # GAN loss
        pred_g_fake = self.netD(x_samples_image)
        if self.opt['train']['gan_type'] == 'gan':
            l_back_gan = self.l_gan_w * self.cri_gan(pred_g_fake, True)
        elif self.opt['train']['gan_type'] == 'ragan':
            pred_d_real = self.netD(gt).detach()
            l_back_gan = self.l_gan_w * (self.cri_gan(pred_d_real - torch.mean(pred_g_fake), False) + self.cri_gan(
                pred_g_fake - torch.mean(pred_d_real), True)) / 2

        # # SoftSkel loss
        # if self.train_opt['lambda_SoftSkelLoss'] > 0:
        #     SoftSkel_weight = self.train_opt['lambda_SoftSkelLoss'] if self.train_opt['lambda_SoftSkelLoss'] else 0
        #     l_softSkel = SoftSkel_weight * self.SoftSkelLoss(gt, x_samples_image)
        # else:
        #     l_softSkel = torch.tensor(0)
        #
        # # TV loss
        # if self.train_opt['lambda_TVLoss'] > 0:
        #     TV_weight = self.train_opt['lambda_TVLoss'] if self.train_opt['lambda_TVLoss'] else 0
        #     l_tv = TV_weight * self.TVLoss(x_samples_image)
        # else:
        #     l_tv = torch.tensor(0)
        #
        # # Sparse loss
        # if self.train_opt['lambda_SparseLoss'] > 0:
        #     Sparse_weight = self.train_opt['lambda_SparseLoss'] if self.train_opt['lambda_SparseLoss'] else 0
        #     l_sparse = Sparse_weight * self.SparseLoss(x_samples_image)
        # else:
        #     l_sparse = torch.tensor(0)

        # FFT loss
        if self.train_opt['lambda_FFTLoss'] > 0:
            FFT_weight = self.train_opt['lambda_FFTLoss'] if self.train_opt['lambda_FFTLoss'] else 0
            l_fft = FFT_weight * self.FrequencyLoss(pred, gt)
        else:
            l_fft = torch.tensor(0)

        return l_back_rec, l_back_fea, l_back_gan,  l_fft

    def optimize_parameters(self, step):
        for p in self.netD.parameters():
            p.requires_grad = False

        self.optimizer_C.zero_grad()

        # forward downscaling
        self.input = self.real_L
        if self.opt['network_C']['which_model_C'] == 'HATpp':
            self.output = self.netC(self.input, size=self.real_L.shape[2:5])
        else:
            self.output = self.netC(x=self.input)

        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            l_back_rec, l_back_fea, l_back_gan, l_fft = self.loss_backward(pred=self.output, gt=self.real_H)

            # total loss
            loss = l_back_rec + l_back_fea + l_back_gan  + l_fft
            loss.backward()

            # gradient clipping
            if self.train_opt['gradient_clipping']:
                nn.utils.clip_grad_norm_(self.netC.parameters(), self.train_opt['gradient_clipping'])

            self.optimizer_C.step()
        # D
        for p in self.netD.parameters():
            p.requires_grad = True

        self.optimizer_D.zero_grad()
        l_d_total = 0
        pred_d_real = self.netD(self.real_H)
        pred_d_fake = self.netD(self.output.detach())
        if self.opt['train']['gan_type'] == 'gan':
            l_d_real = self.cri_gan(pred_d_real, True)
            l_d_fake = self.cri_gan(pred_d_fake, False)
            l_d_total = l_d_real + l_d_fake
        elif self.opt['train']['gan_type'] == 'ragan':
            l_d_real = self.cri_gan(pred_d_real - torch.mean(pred_d_fake), True)
            l_d_fake = self.cri_gan(pred_d_fake - torch.mean(pred_d_real), False)
            l_d_total = (l_d_real + l_d_fake) / 2

        l_d_total.backward()
        self.optimizer_D.step()

        # set log
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            self.log_dict['l_back_rec'] = l_back_rec.item()
            self.log_dict['l_back_fea'] = l_back_fea.item()
            self.log_dict['l_back_gan'] = l_back_gan.item()
            # if self.train_opt['lambda_SoftSkelLoss']:
            #     self.log_dict['l_softSkel'] = l_softSkel.item()
            # if self.train_opt['lambda_TVLoss']:
            #     self.log_dict['l_tv'] = l_tv.item()
            # if self.train_opt['lambda_SparseLoss']:
            #     self.log_dict['l_sparse'] = l_sparse.item()
            if self.train_opt['lambda_FFTLoss'] > 0:
                self.log_dict['l_fft'] = l_fft.item()
        self.log_dict['l_d'] = l_d_total.item()

    def test(self):
        self.input = self.real_L

        self.netC.eval()
        with torch.no_grad():
            if self.opt['network_C']['which_model_C'] == 'HATpp':
                self.SR = self.netC(x=self.input, size=self.input.shape[2:5])
            else:
                self.SR = self.netC(x=self.input)

            self.SR = self.Quantization(self.SR)
        self.netC.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['SR'] = self.SR.detach()[0].float().cpu()
        out_dict['LR'] = self.real_L.detach()[0].float().cpu()
        out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def get_train_SR(self):
        fake_sr = self.output.detach().float().cpu()
        return fake_sr

    def print_network(self):
        s, n = self.get_network_description(self.netC)
        if isinstance(self.netC, nn.DataParallel) or isinstance(self.netC, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netC.__class__.__name__,
                                             self.netC.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netC.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_C = self.opt['path']['pretrain_model_C']
        if load_path_C is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_C))
            self.load_network(load_path_C, self.netC, self.opt['path']['strict_load'])

    def load_D(self):
        load_path_D = self.opt['path']['pretrain_model_D']
        if load_path_D is not None:
            logger.info('Loading model for D [{:s}] ...'.format(load_path_D))
            self.load_network(load_path_D, self.netD, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netC, 'C', iter_label)
        self.save_network(self.netD, 'D', iter_label)

    def feed_real_data(self, data):
        self.real_L = data['LQ'].to(self.device)  # LQ

    def get_current_visuals_LR(self):
        out_dict = OrderedDict()
        out_dict['SR'] = self.SR.detach()[0].float().cpu()
        return out_dict
