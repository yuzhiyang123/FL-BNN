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
from tqdm import trange
import random
from client_binary import Client

class Server():
    def __init__(self, args):
        self.clients = []
        # args = parser.parse_args()
        save_path = os.path.join(args.results_dir, args.save)
        setup_logging(os.path.join(save_path, 'log.txt'))
        results_file = os.path.join(save_path, 'results.%s')
        self.results = ResultsLog(results_file % 'csv', results_file % 'html')
        model = models.__dict__[args.model]
        default_transform = {
            'train': get_transform(args.dataset,
                               input_size=args.input_size, augment=True),
            'eval': get_transform(args.dataset,
                              input_size=args.input_size, augment=False)
        }
        transform = getattr(model, 'input_transform', default_transform)
        train_data = get_dataset(args.dataset, 'train', transform['train'], distribution='noniid', numclients=args.numclients, dataset_path=args.datano)
        if args.numclients == 1:
            train_data=[torch.utils.data.ConcatDataset(train_data)]
        val_data = torch.utils.data.ConcatDataset(get_dataset(args.dataset, 'val', transform['eval'], distribution=None, numclients=1, dataset_path=args.datano))
        self.numclients = args.numclients
        self.alg = args.serveralg
        # self.model_1 = torch.load('model_20.pth')
        # print(self.model_1)
        self.model = models.__dict__[args.model]
        model_config = {'input_size': args.input_size, 'dataset': args.dataset}

        if args.model_config is not '':
            model_config = dict(model_config, **literal_eval(args.model_config))

        self.model = self.model(**model_config)
        # self.model.load_state_dict(self.model_1.state_dict())
        self.criterion = getattr(model, 'criterion', nn.CrossEntropyLoss)()
        self.criterion.type(args.type)
        self.model.type(args.type)
        # proportion = 1/178
        proportion = 1/args.numclients
        torch.cuda.set_device(args.gpus[0])
        for i in range(args.numclients):
            # if i == 36:
            #     proportion = 1/89
            # elif i == 71:
            #     proportion = 1/44.5
            # self.clients.append(Client(i, train_data[i], parser, proportion, self.model))
            self.clients.append(Client(i, train_data[i], args, proportion, self.model))
        self.val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=1024, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        self.args = args
        self.device = []
        for i in range(1):
            self.device.append(torch.device(args.gpus[i]))
        print("Initialized")

    def val(self, val_loader, epoch=0):
        torch.cuda.set_device(self.args.gpus[0])
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

            loss = self.criterion(output, target_var.to(dtype=torch.int64))
            if type(output) is list:
                output = output[0]

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 2))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

        return losses.avg, top1.avg, top5.avg

    def copy_to_full(self):
        # torch.save(self.model, 'model.pth')
        self.args.alpha = 1
        self.args.workmode = 'fullfull'
        # model_tmp = models.femnistnet().cuda()
        # model_tmp.load_state_dict(copy.deepcopy(self.model.state_dict()))
        # self.model = model_tmp
        # for c in self.clients:
        #     c.model = models.femnistnet().cuda()
        #     c.model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        #     c.optimizer = torch.optim.Adam(c.model.parameters(), lr=3e-3)

    def train_epoch(self, epoch, per=None):
        if per is None:
            per=self.numclients
        ppp=per/self.numclients
        sigma = self.args.alpha
        mode = self.args.workmode
        flag = True
        trainloss = AverageMeter()
        traintop1 = AverageMeter()
        traintop5 = AverageMeter()
        valloss = AverageMeter()
        valtop1 = AverageMeter()
        valtop5 = AverageMeter()
        torch.cuda.set_device(self.args.gpus[0])
        for i in range(self.numclients):
        # for i in trange(self.numclients):
            client = self.clients[i]
            loss, top1, top5 = client.train_epoch(epoch)
            trainloss.update(loss)
            traintop1.update(top1)
            traintop5.update(top5)
        if True:
            print("Server epoch", epoch+1, "start")
            with torch.no_grad():
                # for client in self.clients:
                for i in random.sample(range(self.numclients), per):
                    client = self.clients[i]
                    client.precom(mode=mode)
                    if flag:
                        self.model.load_state_dict(copy.deepcopy(client.model.state_dict()))
                        for cv in self.model.parameters():
                            cv.data *= client.proportion/ppp
                        flag = False
                    else:
                        for cv, ccv in zip(self.model.parameters(), client.model.parameters()):
                            cv.data += (ccv.data*client.proportion/ppp).to(self.device[0])
            for client in self.clients:
                client.localupdate(self.model.state_dict(), sigma=sigma, mode=mode)
            for cv in self.model.parameters():
                if hasattr(cv, 'org'):
                    cv.org.copy_(cv.data)
        valloss, valtop1, valtop5 = self.val(self.val_loader, epoch)
        self.results.add(epoch=epoch + 1, train_loss=trainloss.avg, val_loss=valloss,
                    train_error1=100 - traintop1.avg, val_error1=100 - valtop1,
                    train_error5=100 - traintop5.avg, val_error5=100 - valtop5)
        self.results.save()
        '''
        logging.info('\n Epoch: {0}\t'
                     'Training Loss {train_loss:.4f} \t'
                     'Training Prec@1 {train_prec1:.3f} \t'
                     'Training Prec@5 {train_prec5:.3f} \t'
                     'Validation Loss {val_loss:.4f} \t'
                     'Validation Prec@1 {val_prec1:.3f} \t'
                     'Validation Prec@5 {val_prec5:.3f} \n'
                     .format(epoch + 1, train_loss=trainloss.avg, val_loss=valloss,
                             train_prec1=traintop1.avg, val_prec1=valtop1,
                             train_prec5=traintop5.avg, val_prec5=valtop5))
        if epoch == 19:
            torch.save(self.model, 'model_20.pth')
        elif epoch == 39:
            torch.save(self.model, 'model_40.pth')
        elif epoch == 59:
            torch.save(self.model, 'model_60.pth')
        '''
