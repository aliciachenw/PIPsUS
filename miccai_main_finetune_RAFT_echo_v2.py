from __future__ import print_function, division
import sys

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from baseline.RAFT.core.raft import RAFT
# import baseline.RAFT.evaluate as evaluate
from ultrasound.echodata import EchoUSDataset

from torch.utils.tensorboard import SummaryWriter

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000
def sequence_loss(flow_preds, flow_teachers, image1, image2, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    # flow_gt = flow_teachers[-1]
    
    # exlude invalid pixels and extremely large diplacements
    # image1: B x 3 x H x W
    # image2: B x 3 x H x W
    _,_,h,w = image1.size()
    image1 = image1.float()
    image2 = image2.float()
    image1 = image1 / 255.0
    image2 = image2 / 255.0
    indexes = np.empty((h, w, 2))
    x_values = np.repeat(np.reshape(np.arange(w), (1, w)), h, axis=0)
    y_values = np.repeat(np.reshape(np.arange(h), (h, 1)), w, axis=1)

    indexes[:, :, 0] = x_values
    indexes[:, :, 1] = y_values
    grids = torch.from_numpy(indexes).float().cuda().unsqueeze(0).repeat(image1.size(0),1,1,1)
    # print(torch.max(grids), torch.min(grids))

    loss_func = nn.MSELoss()
    l1_loss = nn.L1Loss()
    # for i in range(n_predictions):
    #     i_weight = gamma**(n_predictions - i - 1)

    #     flow_loss += l1_loss(flow_preds[i], flow_teachers[i]) * i_weight / 256 / 256
        
    B,C,H,W = image1.shape
    xx = torch.arange(0, W).view(1 ,-1).repeat(H ,1)
    yy = torch.arange(0, H).view(-1 ,1).repeat(1 ,W)
    xx = xx.reshape(1 ,1 ,H,W).repeat(B ,1 ,1 ,1)
    yy = yy.reshape(1 ,1 ,H,W).repeat(B ,1 ,1 ,1)
    grid = torch.cat((xx ,yy) ,1).float().cuda()
    vgrid = torch.autograd.Variable(grid) + flow_preds[-1]

    vgrid[: ,0 ,: ,:] = 2.0 *vgrid[: ,0 ,: ,:].clone() / max( W -1 ,1 ) -1.0
    vgrid[: ,1 ,: ,:] = 2.0 *vgrid[: ,1 ,: ,:].clone() / max( H -1 ,1 ) -1.0

    vgrid = vgrid.permute(0 ,2 ,3 ,1)
    image1_wrap = F.grid_sample(image2, vgrid)
    flow_loss += l1_loss(image1_wrap, image1)
    metrics = {
    }

    return flow_loss, metrics


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
    #     pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    scheduler = None
    return optimizer, scheduler
    

class Logger:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def train(args):

    model = RAFT(args)
    # teacher = RAFT(args)
    print("Parameter Count: %d" % count_parameters(model))
    model.load_state_dict(torch.load('./baseline/RAFT/models/raft-kitti.pth'), strict=False)
    # teacher.load_state_dict(torch.load('./baseline/RAFT/models/raft-kitti.pth'), strict=False)
    batch_size = args.batch_size
    model.cuda()
    model.train()
    # teacher.cuda()
    # teacher.eval()

    dataset_t = EchoUSDataset('train', (256, 256), use_mini=True)
    dataset_v = EchoUSDataset('val', (256, 256), use_mini=True)
    dataloader_t = DataLoader(dataset_t, batch_size=1, shuffle=True, num_workers=0, drop_last=False)
    dataloader_v = DataLoader(dataset_v, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    # logger = Logger(model, scheduler)

    VAL_FREQ = 5000
    add_noise = True

    best_val_loss = 1000000
    for epoch in range(args.max_epoch):
        model.train()
        for i, data in enumerate(dataloader_t):
            video = data['rgbs'] # B,S,H,W,C
            video_length = video.shape[1]
            vid_loss = 0
            for j in range(0, video_length - 3 - batch_size, batch_size):
                optimizer.zero_grad()
                image1 = video[0,j:j+batch_size].permute(0,3,1,2).cuda()
                image2 = video[0,j+3:j+3+batch_size].permute(0,3,1,2).cuda()

                # if args.add_noise:
                #     stdv = np.random.uniform(0.0, 5.0)
                #     image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                #     image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

                flow_predictions = model(image1, image2, iters=args.iters)            
                # flow_teacher = teacher(image1, image2, iters=args.iters)
                flow_teacher = None
                loss, metrics = sequence_loss(flow_predictions, flow_teacher, image1, image2, args.gamma)
                vid_loss += loss.item()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)                
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                
                scaler.step(optimizer)
                # scheduler.step()
                scaler.update()
            # vid_loss /= video_length
            if i % 5 == 0 or i == len(dataloader_t)-1:
                print('epoch: %d, data: %d/%d, loss: %.4f' % (epoch, i, len(dataloader_t), vid_loss))
        PATH = 'checkpoints/raftecho_v2/%d_%s.pth' % (epoch+1, args.name)
        torch.save(model.state_dict(), PATH)

        # validation
        with torch.no_grad():
            model.eval()
            val_loss = 0
            for i, data in enumerate(dataloader_v):
                video = data['rgbs']
                video_length = video.shape[1]
                vid_loss = 0
                for j in range(0, video_length - 3 - batch_size, batch_size):
                    image1 = video[0,j:j+batch_size].permute(0,3,1,2).cuda()
                    image2 = video[0,j+3:j+3+batch_size].permute(0,3,1,2).cuda()
                    # print(image1.shape, image2.shape)
                    flow_predictions = model(image1, image2, iters=args.iters)
                    # flow_teacher = teacher(image1, image2, iters=args.iters)
                    flow_teacher = None
                    loss, metrics = sequence_loss(flow_predictions, flow_teacher, image1, image2, args.gamma)
                    vid_loss += loss.item()
                # vid_loss /= video_length
                val_loss += vid_loss
            val_loss /= len(dataloader_v)
            print('epoch: %d, val_loss: %.4f' % (epoch, val_loss))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                PATH = 'checkpoints/raftecho_v2/bestval_%d_%s.pth' % (epoch+1, args.name)
                torch.save(model.state_dict(), PATH)
        
        total_steps += 1


    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[256, 256])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=6)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    parser.add_argument('--max_epoch', type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train(args)