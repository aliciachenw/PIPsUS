import numpy as np
from nets.pips2vis import Pips
import torch
import torch.nn.functional as F

DEVICE = 'cuda'



def load_pips2_model():
    model = Pips()
    # todo: load checkpoint
    checkpoint_path='./reference_model/model-000200000.pth'
    print('loading checkpoint from', checkpoint_path)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    return model
    

def pips2_tracking(model, video, start_points, iters=16):
    # using raft optical flow to track keypoints
    video_length, C, H, W = video.shape

    model.eval()
    model.to(DEVICE)

    # prep data
    traj0 = np.expand_dims(start_points, axis=0)
    traj0 = np.repeat(traj0, video_length, axis=0) # S x N x 2
    traj0 = np.expand_dims(traj0, axis=0) # 1 x S x N x 2
    traj0 = torch.from_numpy(traj0).float().to(DEVICE)
    video = video.unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        # inference
        preds, _, valids, _ = model(traj0, video, iters=iters, feat_init=None, beautify=True)
    
    trajs = preds[-1].squeeze(0)  # last prediction is the pseudo ground truth
    valids = valids.squeeze(0)
    return trajs.cpu().numpy(), valids.cpu().numpy()



def generate_pseudo_gt_w_pips2(pips2_model, video, seq_length=20):
    pass




