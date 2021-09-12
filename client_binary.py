import argparse
import os
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import models
from torch.autograd import Variable
from data import get_dataset
from preprocess import get_transform
from utils import *
from datetime import datetime
from ast import literal_eval
from torchvision.utils import save_image
import copy
from scipy.stats import norm

class Client():
    def __init__(self, clientid, train_data, args, proportion=0.01, init_model=None):
        self.proportion = proportion
        self.args = args
        self.clientid = clientid
        save_path = os.path.join(self.args.results_dir, self.args.save)

#        logging.info("Client %d: saving to %s", self.clientid, save_path)
#        logging.debug("Client %d: run arguments: %s", self.clientid, self.args)

        # create model
#        logging.info("Client %d: creating model %s", self.clientid, self.args.model)
        model = models.__dict__[self.args.model]
        model_config = {'input_size': self.args.input_size, 'dataset': self.args.dataset}

        if self.args.model_config is not '':
            model_config = dict(model_config, **literal_eval(self.args.model_config))

        model = model(**model_config)
        logging.info("Client %d: created model with configuration: %s", self.clientid, model_config)

        num_parameters = sum([l.nelement() for l in model.parameters()])
        logging.info("number of parameters: %d", num_parameters)

        # Data loading code
        default_transform = {
            'train': get_transform(self.args.dataset,
                                input_size=self.args.input_size, augment=True),
            'eval': get_transform(self.args.dataset,
                                input_size=self.args.input_size, augment=False)
        }
        transform = getattr(model, 'input_transform', default_transform)
        regime = getattr(model, 'regime', {0: {'optimizer': self.args.optimizer,
                                           'lr': self.args.lr,
                                           'momentum': self.args.momentum,
                                           'weight_decay': self.args.weight_decay}})
        # define loss function (criterion) and optimizer
        self.criterion = getattr(model, 'criterion', nn.CrossEntropyLoss)()
        self.criterion.type(self.args.type)
        model.type(self.args.type)
        self.train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.args.batch_size, shuffle=True,
            num_workers=self.args.workers, pin_memory=True)

        self.optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr)

#        logging.info('training regime: %s', regime)

        self.model = model
        with torch.no_grad():
            if init_model is not None:
                for (p, p_) in zip(self.model.parameters(), init_model.parameters()):
                    p.copy_(p_)
        self.regime = regime
        self.save_path = save_path

    def train(self, epoch):
        # switch to train mode
        self.model.train()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        for i, (inputs, target) in enumerate(self.train_loader):
            if target.size(0) == 1:
                break
            target = target.cuda()
            input_var = Variable(inputs.type(self.args.type), requires_grad=False)
            target_var = Variable(target)
            # compute output
            output = self.model(input_var)
            
            loss = self.criterion(output, target_var.to(dtype=torch.int64))
            if type(output) is list:
                output = output[0]

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 2))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # compute gradient
            self.optimizer.zero_grad()
            loss.backward()
            for p in self.model.parameters():
                if hasattr(p,'org'):
                    p.data.copy_(p.org)
            self.optimizer.step()
            for p in self.model.parameters():
                if hasattr(p,'org'):
                    p.org.copy_(p.data.clamp(-1,1))
            # if i % self.args.print_freq == 0:
                # logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                              #'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                              #'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                              #'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                              #epoch, i, len(self.train_loader),
                              #phase='TRAINING',
                              #loss=losses, top1=top1, top5=top5))

        return losses.avg, top1.avg, top5.avg

    def train_epoch(self, epoch):
        self.optimizer = adjust_optimizer(self.optimizer, epoch, self.regime)
        return self.train(epoch)

    def val(self, val_loader, epoch=0):
        self.model.eval()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        for i, (inputs, target) in enumerate(val_loader):
            target = target.cuda()

            with torch.no_grad():
                input_var = Variable(inputs.type(self.args.type))
                target_var = Variable(target)
                # compute output
                output = self.model(input_var)

            loss = self.criterion(output, target_var)
            if type(output) is list:
                output = output[0]

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 2))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

        return losses.avg, top1.avg, top5.avg

    def localupdate(self, param, sigma=0.2, mode='fullfull',  N=100):
        self.model.load_state_dict(copy.deepcopy(param))
        for cv in self.model.parameters():
            if hasattr(cv, 'org'):
                if mode == 'binfull':
                    cv.org.mul_(1-sigma).add_(cv.data.mul_(sigma))
                elif mode == 'binfullnew':
                    if self.proportion == 0.01:
                        cv.data.mul_(cv.org.sign()).add_(0.9899).log_().mul_(1.3758).add_(-0.0261)
                    elif self.proportion == 0.02:
                        cv.data.mul_(cv.org.sign()).add_(0.9820).log_().mul_(1.3301).add_(-0.0086)
                    elif self.proportion == 0.005:
                        cv.data.mul_(cv.org.sign()).add_(0.9955).log_().mul_(1.4156).add_(-0.0039)
                    elif self.proportion == 1/44 or self.proportion == 1/44.5:
                        cv.data.mul_(cv.org.sign()).add_(0.9797).log_().mul_(1.3212).add_(-0.0352)
                    elif self.proportion == 1/178:
                        cv.data.mul_(cv.org.sign()).add_(0.9943).log_().mul_(1.4074).add_(-0.0352)
                    elif self.proportion == 1/89:
                        cv.data.mul_(cv.org.sign()).add_(0.9887).log_().mul_(1.3683).add_(-0.0231)
                    elif self.proportion == 0.1:
                        cv.data.mul_(cv.org.sign()).add_(0.9652).log_().mul_(1.1717).add_(0.0998)
                    cv.org.mul_(cv.data).mul_(sigma).clamp_(-1,1)
                elif mode == 'fullfull':
                    cv.org.copy_(cv.data)
                elif mode == 'binbin':
                    cv.org.copy_(cv.data)
                    maxtheta=torch.max(cv.org.abs())
                    cv.org.mul_(1.49/maxtheta).round_().div_(1.49)
                    # cv.org.mul_(1-sigma).add_(cv.data.sign_().mul_(sigma))
                else:
                    print("Undefined mode!")

    def precom(self, mode='fullfull'):
        for cv in self.model.parameters():
            if hasattr(cv, 'org'):
                if mode == 'binfull' or mode == 'binfullnew' or mode == 'binbin':
                    # cv.data.sign_()
                    maxtheta=torch.max(cv.data.abs())
                    cv.data.mul_(1.49/maxtheta).round_()
                elif mode == 'fullfull':
                    cv.data.copy_(cv.org)
                else:
                    print("Undefined mode!")
