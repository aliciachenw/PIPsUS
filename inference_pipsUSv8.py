import numpy as np
from nets.pipsUS_v8 import PipsUS
import torch
import torch.nn.functional as F

DEVICE = 'cuda'


def load_pipsUSv8_model(args):
    model = PipsUS(stride=8)
    model.init_realtime_delta()

    # todo: load checkpoint
    if args.data in ['train', 'valid', 'test', 'artificial', 'inplane']:
        checkpoint_path = 'checkpoints/pipsUSv8_i6_S5_size256_256_kpsift_lr1e-4_A_Feb25_finetune/best_val-000000044.pth'
    elif args.data in ['echo', 'echo_artificial']:
        checkpoint_path = 'checkpoints/pipsUScorrMICCAI_echo_i6_S5_size256_256_kpsift_lr1e-4_A_Feb27_finetune/best_val-000000050.pth'
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    return model
    
def pipsUSv8_tracking(model, video, start_points, his_seqlen=5, iters=6):
    # using raft optical flow to track keypoints
    video_length, C, H, W = video.shape

    video = video.float()
    model.eval()
    model.to(DEVICE)
    video = video.to(DEVICE)

    # prep data
    init_rgbs = video[0:1].repeat(his_seqlen,1,1,1).unsqueeze(0).float() # 1 x S x C x H x W

    traj0 = np.expand_dims(start_points, axis=0)
    traj0 = np.repeat(traj0, his_seqlen, axis=0) # S x N x 2
    traj0 = np.expand_dims(traj0, axis=0) # 1 x S x N x 2
    traj0 = torch.from_numpy(traj0).float().to(DEVICE)
    
    # get sub seq
    # number of points
    N = traj0.shape[2]
    traj_pre = traj0.clone()
    trajs_e = torch.zeros((1, video_length, N, 2)).to(DEVICE)
    trajs_e[0, 0:1] = traj0[0,-1:]
    model.eval()
    prev = init_rgbs.clone()

    # feat1 = model.init_feat(traj0, init_rgbs)
    # feat_bank = [feat1.clone(), feat1.clone(), feat1.clone(), feat1.clone(), feat1.clone(), feat1.clone(), feat1.clone(), feat1.clone()]
    with torch.no_grad():
        for i in range(1, video_length):            
            curr = video[i].unsqueeze(0) # 1 x C x H x W
            preds_coords, _, _ = model(traj_pre, prev, curr, iters=iters, beautify=True)

            # preds_coords, _, new_feat = model(traj_pre, prev, curr, iters=iters, feat_pre=[feat_bank[0], feat_bank[-4], feat_bank[-2]], beautify=True, return_feat=True)
            pred_point = preds_coords[-1] # 1 x N x 2
            trajs_e[0,i] = pred_point[0]

            # update traj_pre
            traj_pre = torch.cat([trajs_e[0,0:1], traj_pre[0,2:], pred_point], dim=0).unsqueeze(0)
            prev = torch.cat([init_rgbs[:,0:1], prev[:,2:], curr.unsqueeze(0)], dim=1)
            # feat_bank.pop(1)
            # feat_bank.append(new_feat.clone())

    valids = np.ones((video_length, N))
    return trajs_e[0].cpu().numpy(), valids

