import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import ReconstructionLoss, PerceptualLoss
from models.modules.Quantization import Quantization

logger = logging.getLogger('base')


class SingleModel(BaseModel):
    def __init__(self, opt):
        super(SingleModel, self).__init__(opt)

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
            self.netC.train()

            # loss
            self.Reconstruction_back = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_back'])

            # optimizers
            wd_C = train_opt['weight_decay_C'] if train_opt['weight_decay_C'] else 0
            self.optimizer_C = torch.optim.Adam(self.netC.parameters(), lr=train_opt['lr_C'], weight_decay=wd_C,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_C)

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

    def loss(self, x, y):
        l_back_rec = self.train_opt['lambda_rec_back'] * self.Reconstruction_back(x, y)
        return l_back_rec

    def optimize_parameters(self, step):
        self.optimizer_C.zero_grad()

        # forward downscaling
        self.input = self.real_L
        if self.opt['network_C']['which_model_C'] == 'HATpp':
            self.output = self.netC(self.input, size=self.real_L.shape[2:5])
        else:
            self.output = self.netC(x=self.input)

        l_re = self.loss(self.output, self.real_H)

        # total loss
        loss = l_re
        loss.backward()

        # gradient clipping
        if self.train_opt['gradient_clipping']:
            nn.utils.clip_grad_norm_(self.netC.parameters(), self.train_opt['gradient_clipping'])

        self.optimizer_C.step()

        # set log
        self.log_dict['l_re'] = l_re.item()
        # self.log_dict['l_fea'] = l_back_fea.item()

    def test(self):
        self.input = self.real_L

        self.netC.eval()
        with torch.no_grad():
            if self.opt['network_C']['which_model_C'] == 'HAT':
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
        Denoise = self.output.detach().float().cpu()
        Denoise = self.output.detach().float().cpu()
        return Denoise

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

    def save(self, iter_label):
        self.save_network(self.netC, 'C', iter_label)

    def feed_real_data(self, data):
        self.real_L = data['LQ'].to(self.device)  # LQ

    def get_current_visuals_LR(self):
        out_dict = OrderedDict()
        out_dict['SR'] = self.SR.detach()[0].float().cpu()
        return out_dict
