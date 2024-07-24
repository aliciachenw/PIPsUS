"""
more like pips2, only use first, -2, -4 frame, don't use previous motion flow
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.basic
import utils.samp
import utils.misc
from torch import nn
from nets.pips2 import DeltaBlock

class Conv1dPad(nn.Module):
    """
    nn.Conv1d with auto-computed padding ("same" padding)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(Conv1dPad, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            groups=self.groups)

    def forward(self, x):
        net = x
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        net = self.conv(net)
        return net
    
class ResidualBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, use_norm, use_do, is_first_block=False):
        super(ResidualBlock1d, self).__init__()
        
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.stride = 1
        self.is_first_block = is_first_block
        self.use_norm = use_norm
        self.use_do = use_do

        self.norm1 = nn.InstanceNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = Conv1dPad(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=self.stride,
            groups=self.groups)

        self.norm2 = nn.InstanceNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = Conv1dPad(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=1,
            groups=self.groups)

    def forward(self, x):
        
        identity = x
        
        out = x
        if not self.is_first_block:
            if self.use_norm:
                out = self.norm1(out)
            out = self.relu1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)
        
        if self.use_norm:
            out = self.norm2(out)
        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)
            
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1,-2)
            ch1 = (self.out_channels-self.in_channels)//2
            ch2 = self.out_channels-self.in_channels-ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1,-2)
        
        out += identity
        return out


class ResidualBlock2d(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock2d, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride, padding_mode='zeros')
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, padding_mode='zeros')
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)


    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)
    
class BasicEncoder(nn.Module):
    def __init__(self, input_dim=3, output_dim=128, stride=8, norm_fn='batch', dropout=0.0):
        super(BasicEncoder, self).__init__()
        self.stride = stride
        self.norm_fn = norm_fn

        self.in_planes = 64
        
        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=self.in_planes)
            self.norm2 = nn.GroupNorm(num_groups=8, num_channels=output_dim*2)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.InstanceNorm2d(self.in_planes)
            self.norm2 = nn.InstanceNorm2d(output_dim*2)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(self.in_planes)
            self.norm2 = nn.InstanceNorm2d(output_dim*2)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()
            
        self.conv1 = nn.Conv2d(input_dim, self.in_planes, kernel_size=7, stride=2, padding=3, padding_mode='zeros')
        self.relu1 = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64,  stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)
        self.layer4 = self._make_layer(128, stride=2)

        self.conv2 = nn.Conv2d(128+128+96+64, output_dim*2, kernel_size=3, padding=1, padding_mode='zeros')
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(output_dim*2, output_dim, kernel_size=1)
        
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.InstanceNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock2d(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock2d(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):

        _, _, H, W = x.shape
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        a = self.layer1(x)
        b = self.layer2(a)
        c = self.layer3(b)
        d = self.layer4(c)
        a = F.interpolate(a, (H//self.stride, W//self.stride), mode='bilinear', align_corners=True)
        b = F.interpolate(b, (H//self.stride, W//self.stride), mode='bilinear', align_corners=True)
        c = F.interpolate(c, (H//self.stride, W//self.stride), mode='bilinear', align_corners=True)
        d = F.interpolate(d, (H//self.stride, W//self.stride), mode='bilinear', align_corners=True)

        # x = torch.cat([a,b,c,d], dim=1)
        # return x
    
        x = self.conv2(torch.cat([a,b,c,d], dim=1))
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)
        return x

class DeltaBlockRealtime(nn.Module): # use implicit euler: y_{k+1} = y_k + Delta(y_{k+1})
    def __init__(self, latent_dim=128, hidden_dim=128, corr_levels=4, corr_radius=3, seq_len=5):
        super(DeltaBlockRealtime, self).__init__()
        
        kitchen_dim = (corr_levels * (2*corr_radius + 1)**2)   # concatenate correlation map (corr level * patch size channels) + feature latent dim + x, y coordinates

        self.hidden_dim = hidden_dim
        self.corr_radius = corr_radius
        in_channels = kitchen_dim
        base_filters = 128
        self.n_block = 8
        self.kernel_size = 3
        self.groups = 1
        self.use_norm = False
        self.use_do = False

        self.increasefilter_gap = 2 

        self.first_block_conv = Conv1dPad(in_channels=in_channels, out_channels=base_filters, kernel_size=self.kernel_size, stride=1)
        # self.first_block_norm = nn.InstanceNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        out_channels = base_filters
        
        self.S = seq_len

        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):

            if i_block == 0:
                is_first_block = True
            else:
                is_first_block = False

            if is_first_block:
                in_channels = base_filters
                out_channels = in_channels
            else:
                in_channels = int(base_filters*2**((i_block-1)//self.increasefilter_gap))
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels
            
            tmp_block = ResidualBlock1d(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=self.kernel_size, 
                stride=1, 
                groups=self.groups, 
                use_norm=self.use_norm, 
                use_do=self.use_do, 
                is_first_block=is_first_block)
            self.basicblock_list.append(tmp_block)

        # self.final_norm = nn.InstanceNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        self.dense = nn.Linear(out_channels * self.S, 2)
            
    def forward(self, fcorr):

        x = fcorr # B, S, LRR
        # conv1d wants channels in the middle
        out = x.permute(0,2,1)
        out = self.first_block_conv(out)
        out = self.first_block_relu(out)
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block] # 1d resblock
            out = net(out)
        out = self.final_relu(out) # B,C,S
        out = out.permute(0,2,1).reshape(out.shape[0], -1) # B,SxC
        delta = self.dense(out) # B, 2

        return delta


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    # go to 0,1 then 0,2 then -1,1
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()
    return img

def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd), indexing='ij')
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)

class CorrBlock:
    def __init__(self, fmaps, S, num_levels=4, radius=4):
        B, C, H, W = fmaps.shape
        self.C, self.H, self.W = C, H, W
        self.S = S
        self.num_levels = num_levels
        self.radius = radius
        self.fmaps_pyramid = []
        self.fmaps_pyramid.append(fmaps.unsqueeze(1).repeat(1,S,1,1,1))
        for i in range(self.num_levels-1):
            fmaps = F.avg_pool2d(fmaps, 2, stride=2) # B,C,H',W'
            self.fmaps_pyramid.append(fmaps.unsqueeze(1).repeat(1,S,1,1,1)) # change to B,S,C,H',W'

    def sample(self, coords): # sample from the corr map
        r = self.radius
        B, N, D = coords.shape # N: number of points to track
        assert(D==2)

        H, W = self.H, self.W
        out_pyramid = []
        for i in range(self.num_levels):
            corrs = self.corrs_pyramid[i] # B,S,N,H,W, for each [,s,n] it is corr map of [t_curr, n] with [t_curr-(S-s), n]
            _, _, _, H, W = corrs.shape
            
            dx = torch.linspace(-r, r, 2*r+1)
            dy = torch.linspace(-r, r, 2*r+1)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing='ij'), axis=-1).to(coords.device) 

            centroid_lvl = coords.reshape(B*N, 1, 1, 2) / 2**i # point coordinate at this level  # B*N,1,1,2
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl # B*N,2*r+1,2*r+1,2
            corrs = bilinear_sampler(corrs.permute(0,2,1,3,4).reshape(B*N, self.S, H, W), coords_lvl) # B*N,S,2*r+1,2*r+1 
            # corrs = corrs.view(B, self.S, N, -1) # B,S,N,RR
            corrs = corrs.reshape(B, N, self.S, -1).permute(0, 2, 1, 3) # B,S,N,RR -> sample corr map (t_curr, t-s) at kp at t_curr
            out_pyramid.append(corrs)

        out = torch.cat(out_pyramid, dim=-1) # B,S,N,LRR*2
        return out.contiguous().float()

    def corr(self, targets): # generate corr map
        B, S, N, C = targets.shape # C: latent_dim
        assert(C==self.C)
        assert(S==self.S)
        self.corrs_pyramid = []
        for fmaps in self.fmaps_pyramid:
            _, _, _, H, W = fmaps.shape # B x S x C x H x W
            fmap2s = fmaps.view(B, S, C, H*W)
            corrs = torch.matmul(targets, fmap2s) # matmul: B x S x N x C, B x S x C x H*W -> B x S x N x H*W
            corrs = corrs.view(B, self.S, N, H, W) 
            corrs = corrs / torch.sqrt(torch.tensor(C).float())
            self.corrs_pyramid.append(corrs)


class PipsUS(nn.Module):
    def __init__(self, stride=8, pips2_pretrain_compatibility=True):

        # NOTE: hacking to reuse previous keepfirst training code
        super(PipsUS, self).__init__()

        self.stride = stride

        self.hidden_dim = hdim = 256
        self.latent_dim = latent_dim = 128
        self.corr_levels = 4
        self.corr_radius = 3
        self.seq_len = 3
        
        self.fnet = BasicEncoder(output_dim=self.latent_dim, norm_fn='instance', dropout=0, stride=stride)
        self.delta_block = DeltaBlock(hidden_dim=self.hidden_dim, corr_levels=self.corr_levels, corr_radius=self.corr_radius)
        self.norm = nn.GroupNorm(1, self.latent_dim)
        self.encoder_freeze = False
        if not pips2_pretrain_compatibility:
            self.init_realtime_delta()


    # trick to load pretrained
    def init_realtime_delta(self):
        self.delta_block = DeltaBlockRealtime(latent_dim=self.latent_dim, hidden_dim=self.hidden_dim, corr_levels=self.corr_levels, corr_radius=self.corr_radius, seq_len=self.seq_len)

    def return_parameter(self):
        if self.encoder_freeze:
            param_list = []
            for param in self.delta_block.parameters():
                param_list.append(param)
        else:
            param_list = list(self.fnet.parameters()) + list(self.delta_block.parameters())
        return param_list

    def freeze_encoder(self, require_grad=True):
        for param in self.fnet.parameters():
            param.requires_grad = require_grad
        self.encoder_freeze = not require_grad

            
    def forward(self, trajs_previous, image_previous, image_curr, iters=3, feat_pre=None, valids=None, sw=None, beautify=False, return_feat=False):
        # trajs_previous: B x seq_len x N x 2
        # image_previous: B x seq_len x C x H x W
        # image_curr: B x 1 x C x H x W

        coords = trajs_previous[:,-1,:,:].clone()/float(self.stride) # assume no motion, B x N x 2, also downsampled to feature map resolution

        # NOTE: hacking to reuse previous keepfirst training code
        trajs_previous = torch.cat((trajs_previous[:,0:1,:,:],trajs_previous[:,-4:-3,:,:],trajs_previous[:,-2:-1,:,:]),dim=1)
        trajs_previous = trajs_previous / float(self.stride)
        image_previous = torch.cat((image_previous[:,0:1,:,:,:],image_previous[:,-4:-3,:,:,:],image_previous[:,-2:-1,:,:,:]),dim=1)


        B,S,N,D = trajs_previous.shape
        assert(D==2)
        assert(S==self.seq_len)

        B,S,C,H,W = image_previous.shape
        image_previous = 2 * (image_previous / 255.0) - 1.0
        image_curr = 2 * (image_curr / 255.0) - 1.0
        assert(C==3)
        assert(S==self.seq_len)
        
        H8 = H//self.stride
        W8 = W//self.stride

        # get the features map
        if feat_pre is None:
            image_previous_ = image_previous.reshape(B*S, C, H, W)
            fmaps_pre = self.fnet(image_previous_)  # first feature map, should be B*S x latent_dim x H8 x W8
            # reshape pre traj
            trajs_previous = trajs_previous.reshape(B*S, N, 2)
            feat_pre = utils.samp.bilinear_sample2d(fmaps_pre, trajs_previous[:,:,0], trajs_previous[:,:,1]).permute(0, 2, 1) # B*S,N,C
            feat_pre = feat_pre.reshape(B,S,N,self.latent_dim) # B, S, N, latent
            # reshape back
            trajs_previous = trajs_previous.reshape(B,S,N,2)# B,S,N,2
        else:
            assert len(feat_pre) == S
            feat_pre = torch.cat(feat_pre, dim=1) # B, S, N, latent

        # fmaps_pre = fmaps_.reshape(B, S, self.latent_dim, H8, W8)
        image_curr_ = image_curr.reshape(B, C, H, W)
        fmaps_curr = self.fnet(image_curr_)  # should be B x latent_dim x H8 x W8
        
        fcorr_fn = CorrBlock(fmaps_curr, S=S, num_levels=self.corr_levels, radius=self.corr_radius)
        
        coord_predictions1 = [] # for loss
        coord_predictions2 = [] # for vis

        coord_predictions2.append(coords.detach() * self.stride)
        

        fcorr_fn.corr(feat_pre) # compute correlation map of current sampled feature with previous feature maps

        for itr in range(iters):
            coords = coords.detach() # B x N x 2 -> we don't want to backprop through this, we only want to learn good delta
            # now we want costs at the current locations
            fcorrs = fcorr_fn.sample(coords) # B,S,N,LRR  # -> sample corr map (t_curr, t-s) at kp at t_curr

            LRR = fcorrs.shape[3]

            # we want everything in the format B*N, S, C
            fcorrs_ = fcorrs.permute(0, 2, 1, 3).reshape(B*N, S, LRR)

            delta_coords_ = self.delta_block(fcorrs_) # B*N, 2 -> learn to optimize delta coord, gradient is generated here

            if beautify and itr > 3*iters//4:
                # this smooths the results a bit, but does not really help perf
                delta_coords_ = delta_coords_ * 0.5

            coords = coords + delta_coords_.reshape(B, N, 2) # B,N,2

            coord_predictions1.append(coords * self.stride)
            coord_predictions2.append(coords * self.stride)
            
        # pause at the end, to make the summs more interpretable
        coord_predictions2.append(coords * self.stride)
        coord_predictions1.append(coords * self.stride)  # already rescale to original size
        new_feat = utils.samp.bilinear_sample2d(fmaps_curr, coords[:,:,0], coords[:,:,1]).permute(0, 2, 1).unsqueeze(1) # B,N,C

        return coord_predictions1, coord_predictions2, new_feat
    


    def init_feat(self, trajs_previous, image_previous):

        trajs_sample = trajs_previous[:,0:1,:,:]//float(self.stride)
        image_previous = image_previous[:,0:1,:,:,:]


        B,S,N,D = trajs_sample.shape
        B,S,C,H,W = image_previous.shape
        image_previous = 2 * (image_previous / 255.0) - 1.0

        # get the features map

        image_previous_ = image_previous.reshape(B*S, C, H, W)
        fmaps_pre = self.fnet(image_previous_)  # first feature map, should be B*S x latent_dim x H8 x W8
        # reshape pre traj
        trajs_sample = trajs_sample.reshape(B*S, N, 2)
        feat_pre = utils.samp.bilinear_sample2d(fmaps_pre, trajs_sample[:,:,0], trajs_sample[:,:,1]).permute(0, 2, 1) # B*S,N,C
        feat_pre = feat_pre.reshape(B,S,N,self.latent_dim) # B, S, N, latent

        return feat_pre