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
import bokeh
from torch.autograd import Variable
from data import get_dataset
from preprocess import get_transform
from utils import *
from datetime import datetime
from ast import literal_eval
from torchvision.utils import save_image
from server_binary import Server
import copy


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ConvNet Training')

parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results',
                    help='results dir')
parser.add_argument('--save', metavar='SAVE', default='',
                    help='saved folder')
parser.add_argument('--dataset', metavar='DATASET', default='imagenet',
                    help='dataset name or folder')
parser.add_argument('--model', '-a', metavar='MODEL', default='alexnet',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: alexnet)')
parser.add_argument('--input_size', type=int, default=None,
                    help='image input size')
parser.add_argument('--model_config', default='',
                    help='additional architecture configuration')
parser.add_argument('--type', default='torch.cuda.FloatTensor',
                    help='type of tensor - e.g torch.cuda.HalfTensor')
parser.add_argument('--gpus', default='0',
                    help='gpus used for training - e.g 0,1,3')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=2500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--optimizer', default='SGD', type=str, metavar='OPT',
                    help='optimizer function used')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', type=str, metavar='FILE',
                    help='evaluate model FILE on validation set')
parser.add_argument('-n', '--numclients', type=int, default=4,
                    help='number of clients')
parser.add_argument('--serveralg', type=str, default='Naive',
                    help='server parameters updating algorithm')
parser.add_argument('--workmode', type=str, default='fullfull',
                    help='system working mode')
parser.add_argument('--alpha', type=float, default=0.2,
                    help='client parameters updating algorithm')


def main():
    args = parser.parse_args()
    if 'cuda' in args.type:
        args.gpus = [int(i) for i in args.gpus.split(',')]
        torch.cuda.set_device(args.gpus[0])
        cudnn.benchmark = True
    else:
        args.gpus = None
    if args.evaluate:
        args.results_dir = '/tmp'
    if args.save is '':
        args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    args.datano = '0'

    n = args.numclients
    __args = []
    # fl
    __args.append(copy.copy(args))
    # cen
    args.save = 'cen'
    args.numclients = 1
    #__args.append(copy.copy(args))
    # BNN
    args.save = 'bnn'
    args.model = args.model+'_binary'
    #__args.append(copy.copy(args))
    # fullfull
    args.save = 'fullfull'
    args.numclients = n
    #__args.append(copy.copy(args))
    # binbinbeta1
    args.save = 'binbinbeta1'
    args.workmode = 'binbin'
    __args.append(copy.copy(args))
    args.save = 'fttq'
    #__args.append(copy.copy(args))
    args.save = 'niid3_1'
    # args.datano = '1'
    #__args.append(copy.copy(args))
    args.save = 'niid3_2'
    # args.datano = '2'
    #__args.append(copy.copy(args))
    args.save = 'niid3_3'
    # args.datano = '3'
    #__args.append(copy.copy(args))
    args.save = 'niid3_4'
    # args.datano = '4'
    #__args.append(copy.copy(args))
    # binbinbeta0.3
    args.save = 'binbinbeta0.3'
    args.alpha = 0.3
    #__args.append(copy.copy(args))
    # binfullbeta0.3
    args.save = 'binfullbeta0.3'
    args.workmode = 'binfull'
    __args.append(copy.copy(args))
    # binfullalpha1.5
    args.save = 'binfullalpha1.75'
    args.workmode = 'binfullnew'
    args.alpha = 1.75
    __args.append(copy.copy(args))

    for args in __args:
        print(args)

    for args in  __args:
        # args.results_dir = 'results_niid_per0.1'
        save_path = os.path.join(args.results_dir, args.save)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        server_ = Server(args)
        for epoch in range(args.start_epoch, args.epochs):
            server_.train_epoch(epoch, per=None)
    '''   
    for args in  __args:
        args.results_dir = 'results_niid_per0.3'
        save_path = os.path.join(args.results_dir, args.save)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        server_ = Server(args)
        for epoch in range(args.start_epoch, args.epochs):
            server_.train_epoch(epoch, per=30)
        
    for args in  __args:
        args.results_dir = 'results_niid_per0.5'
        save_path = os.path.join(args.results_dir, args.save)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        server_ = Server(args)
        for epoch in range(args.start_epoch, args.epochs):
            server_.train_epoch(epoch, per=50)
        # server_.copy_to_full()
        # for epoch in range(60, args.epochs):
        #     server_.train_epoch(epoch)
    '''

if __name__ == '__main__':
    main()
