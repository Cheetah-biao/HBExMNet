import copy
import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import ReconstructionLoss, GANLoss, PerceptualLoss, FrequencyLoss, TVLoss, \
    EdgeLoss, SoftSkelLoss, PearsonCorrelationLoss
from models.modules.Quantization import Quantization

logger = logging.getLogger('base')


class DualstageGanModel(BaseModel):
    def __init__(self, opt):
        super(DualstageGanModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']
        test_opt = opt['test']
        self.train_opt = train_opt
        self.test_opt = test_opt

        self.l_gan_w = train_opt['lambda_gan']
        self.l_fea_w = train_opt['lambda_feature']
        self.l_tv_w = train_opt['lambda_tv']
        self.l_edge_w = train_opt['lambda_edge']
        self.l_skel_w = train_opt['lambda_skel']
        self.l_pearson_w = train_opt['lambda_pearson']

        self.out_channel = opt['Net1']['out_nc']

        self.net1 = networks.define_Net1(opt).to(self.device)
        self.net_sr = networks.define_Net_SR(opt).to(self.device)
        if opt['dist']:
            self.net1 = DistributedDataParallel(self.net1, device_ids=[torch.cuda.current_device()])
            self.net_sr = DistributedDataParallel(self.net_sr, device_ids=[torch.cuda.current_device()])
        else:
            self.net1 = DataParallel(self.net1)
            self.net_sr = DataParallel(self.net_sr)
        # print network
        self.print_network()
        self.load()

        self.Quantization = Quantization()

        if self.is_train:
            self.net1.train()
            self.net_sr.train()
            if self.l_gan_w > 0:
                self.netD = networks.define_D(opt).to(self.device)
                if opt['dist']:
                    self.netD = DistributedDataParallel(self.netD, device_ids=[torch.cuda.current_device()])
                else:
                    self.netD = DataParallel(self.netD)
                self.netD.train()
                self.load_D()
                # GD gan loss
                self.cri_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)

            # loss
            self.l_mip_w = train_opt['lambda_mip'] if train_opt['lambda_mip'] else 0
            self.net1_loss = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_net1'],
                                                   l_mip=self.l_mip_w)
            self.SR1_loss = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_sr'])
            self.FrequencyLoss = FrequencyLoss()
            self.TV_l = TVLoss()
            self.Edge_l = EdgeLoss()
            self.skel_l = SoftSkelLoss()
            self.Pearson_l = PearsonCorrelationLoss()

            # feature loss
            opt_feature = train_opt['perceptual_opt']
            if self.l_fea_w > 0:
                self.Featureloss = PerceptualLoss(
                    layer_weights=opt_feature['layer_weights'],
                    vgg_type='vgg19',
                    use_input_norm=True,
                    range_norm=False,
                    perceptual_weight=1.0,
                    criterion='l1',
                    l_mip=self.l_mip_w,
                ).to(self.device)
        else:
            self.l_fea_w = 0
        # if self.l_gan_w > 0:
        # D_update_ratio and D_init_iters
        self.D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] else 1
        self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0

        # optimizers
        wd_C0 = train_opt['weight_decay_C'] if train_opt['weight_decay_C'] else 0
        self.optimizer_C0 = torch.optim.Adam(self.net1.parameters(), lr=train_opt['lr_C'], weight_decay=wd_C0,
                                             betas=(train_opt['beta1'], train_opt['beta2']))
        self.optimizers.append(self.optimizer_C0)

        wd_C1 = train_opt['weight_decay_C'] if train_opt['weight_decay_C'] else 0
        self.optimizer_C1 = torch.optim.Adam(self.net_sr.parameters(), lr=train_opt['lr_C'], weight_decay=wd_C1,
                                             betas=(train_opt['beta1'], train_opt['beta2']))
        self.optimizers.append(self.optimizer_C1)

        if self.l_gan_w > 0:
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
        self.LSNR = data['LQ'].to(self.device)  # LQ
        self.HSNR = data['HQ'].to(self.device)  # HQ
        self.GT = data['GT'].to(self.device)  # GT

    def loss_net1(self, x, y):
        l_back_rec = self.net1_loss(x, y)
        return l_back_rec

    def feature_loss(self, fake, real):
        l_g_fea = self.l_fea_w * self.Featureloss(fake, real)
        return l_g_fea

    def loss_backward(self, pred, gt):
        # mse loss
        x_samples_image = pred[:, :self.out_channel, :, :]
        l_back_rec = self.train_opt['lambda_rec_back'] * self.SR1_loss(gt, x_samples_image)

        # feature loss
        if self.l_fea_w > 0:
            l_back_fea = self.feature_loss(x_samples_image, gt)
        else:
            l_back_fea = torch.tensor(0)

        # GAN loss
        if self.l_gan_w > 0:
            pred_g_fake = self.netD(x_samples_image)
            if self.opt['train']['gan_type'] == 'gan':
                l_back_gan = self.l_gan_w * self.cri_gan(pred_g_fake, True)
            elif self.opt['train']['gan_type'] == 'ragan':
                pred_d_real = self.netD(gt).detach()
                l_back_gan = self.l_gan_w * (self.cri_gan(pred_d_real - torch.mean(pred_g_fake), False) + self.cri_gan(
                    pred_g_fake - torch.mean(pred_d_real), True)) / 2
        else:
            l_back_gan = torch.tensor(0)

        # FFT loss
        if self.train_opt['lambda_FFTLoss'] > 0:
            FFT_weight = self.train_opt['lambda_FFTLoss'] if self.train_opt['lambda_FFTLoss'] else 0
            l_fft = FFT_weight * self.FrequencyLoss(pred, gt)
        else:
            l_fft = torch.tensor(0)

        return l_back_rec, l_back_fea, l_back_gan, l_fft

    def using_loss(self, w, loss_t):
        if w > 0:
            l = w*loss_t(self.SR, self.GT)
        else:
            l = torch.tensor(0)
        return l

    def optimize_parameters(self, step):
        self.optimizer_C0.zero_grad()
        self.optimizer_C1.zero_grad()
        if self.l_gan_w > 0:
            for p in self.netD.parameters():
                p.requires_grad = False

        # forward downscaling
        self.input = self.LSNR
        self.gt0 = self.HSNR.detach()
        self.output0 = self.net1(x=self.input)
        if self.opt['Net_SR']['which_model'] == 'HAT':
            self.SR = self.net_sr(x=self.output0, size=self.output0.shape[2:5])
        else:
            self.SR = self.net_sr(x=self.output0)

        l_net1 = self.loss_net1(self.output0, self.gt0)
        l_sr, l_sr_fea, l_sr_gan, l_fft = self.loss_backward(self.SR, self.GT)

        l_tv = self.using_loss(self.l_tv_w, self.TV_l)
        l_edge = self.using_loss(self.l_edge_w, self.Edge_l)
        l_skel = self.using_loss(self.l_skel_w, self.skel_l)
        l_pearson = self.using_loss(self.l_pearson_w, self.Pearson_l)

        if self.l_gan_w > 0:
            for p in self.netD.parameters():
                p.requires_grad = False

        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            l_back_rec, l_back_fea, l_back_gan, l_fft = self.loss_backward(pred=self.SR, gt=self.GT)

            # total loss
            l_net1.backward(retain_graph=True)
            # self.optimizer_C0.step()

            loss = 2 * l_net1 + l_back_rec + l_back_fea + l_back_gan + l_fft + l_tv + l_edge + l_skel + l_pearson
            loss.backward()

            # gradient clipping
            if self.train_opt['gradient_clipping']:
                nn.utils.clip_grad_norm_(self.net1.parameters(), self.train_opt['gradient_clipping'])
                nn.utils.clip_grad_norm_(self.net_sr.parameters(), self.train_opt['gradient_clipping'])

            self.optimizer_C0.step()
            self.optimizer_C1.step()

        if self.l_gan_w > 0:
            # D
            for p in self.netD.parameters():
                p.requires_grad = True

            self.optimizer_D.zero_grad()
            l_d_total = 0
            pred_d_real = self.netD(self.GT)
            pred_d_fake = self.netD(self.SR.detach())
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
            self.log_dict['l_net1'] = l_net1.item()
            self.log_dict['l_back_rec'] = l_sr.item()
            if self.l_fea_w > 0:
                self.log_dict['l_back_fea'] = l_sr_fea.item()
            if self.l_gan_w > 0:
                self.log_dict['l_back_gan'] = l_sr_gan.item()
            if self.train_opt['lambda_FFTLoss'] > 0:
                self.log_dict['l_fft'] = l_fft.item()
            if self.l_tv_w > 0:
                self.log_dict['l_tv'] = l_tv.item()
            if self.l_edge_w > 0:
                self.log_dict['l_edge'] = l_edge.item()
            if self.l_skel_w > 0:
                self.log_dict['l_skel'] = l_skel.item()
            if self.l_pearson_w > 0:
                self.log_dict['l_pearson'] = l_pearson.item()
        if self.l_gan_w > 0:
            self.log_dict['l_d'] = l_d_total.item()

        # gradient clipping
        if self.train_opt['gradient_clipping']:
            nn.utils.clip_grad_norm_(self.net1.parameters(), self.train_opt['gradient_clipping'])
            nn.utils.clip_grad_norm_(self.net_sr.parameters(), self.train_opt['gradient_clipping'])

    def test(self):
        self.input = self.LSNR

        self.net1.eval()
        with torch.no_grad():
            self.Net1_img = self.net1(x=self.input)
            if self.opt['Net_SR']['which_model'] == 'HAT':
                self.SR_img = self.net_sr(x=self.Net1_img, size=self.Net1_img.shape[2:5])
            else:
                self.SR_img = self.net_sr(x=self.Net1_img)
            self.Net1_img = self.Quantization(self.Net1_img)
            self.SR_img = self.Quantization(self.SR_img)
        self.net1.train()
        self.net_sr.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['Net1_img'] = self.Net1_img.detach()[0].float().cpu()
        out_dict['SR_img'] = self.SR_img.detach()[0].float().cpu()
        out_dict['LSNR'] = self.LSNR.detach()[0].float().cpu()
        out_dict['GT'] = self.GT.detach()[0].float().cpu()
        return out_dict

    def get_train_SR(self):
        fake_sr = self.SR.detach().float().cpu()
        return fake_sr

    def print_network(self):
        num = 0
        for i in (self.net1, self.net_sr):
            s, n = self.get_network_description(i)
            if isinstance(i, nn.DataParallel) or isinstance(i, DistributedDataParallel):
                net_struc_str = '{} - {}'.format(i.__class__.__name__,
                                                 i.module.__class__.__name__)
            else:
                net_struc_str = '{}'.format(i.__class__.__name__)
            if self.rank <= 0:
                logger.info('Network{:,d} structure: {}, with parameters: {:,d}'.format(num, net_struc_str, n))
                logger.info(s)
            num = num + 1

    def load(self):
        load_path_Net1 = self.opt['path']['pretrain_model_net1']
        if load_path_Net1 is not None:
            logger.info('Loading model for Net1 [{:s}] ...'.format(load_path_Net1))
            self.load_network(load_path_Net1, self.net1, self.opt['path']['strict_load'])

        load_path_SR = self.opt['path']['pretrain_model_SR']
        if load_path_SR is not None:
            logger.info('Loading model for SR [{:s}] ...'.format(load_path_SR))
            self.load_network(load_path_SR, self.net_sr, self.opt['path']['strict_load'])

    def load_D(self):
        load_path_D = self.opt['path']['pretrain_model_D']
        if load_path_D is not None:
            logger.info('Loading model for D [{:s}] ...'.format(load_path_D))
            self.load_network(load_path_D, self.netD, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.net1, 'Net1', iter_label)
        self.save_network(self.net_sr, 'SR', iter_label)
        if self.l_gan_w > 0:
            self.save_network(self.netD, 'D', iter_label)

    def feed_real_data(self, data):
        self.real_L = data['LQ'].to(self.device)  # LQ
