import numpy as np
from nets.pips2 import Pips
import torch
import torch.nn.functional as F

DEVICE = 'cuda'



def load_pips2ncc_model(checkpoint_path='./reference_model/model-000200000.pth'):
    model = Pips(stride=1)
    # todo: load checkpoint
    print('loading checkpoint from', checkpoint_path)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    return model
    

def pips2ncc_tracking(model, video, start_points, iters=16):
    # using raft optical flow to track keypoints
    video_length, C, H, W = video.shape

    model.eval()
    model.to(DEVICE)

    # prep data
    trajs = np.zeros((video_length, start_points.shape[0], 2))
    trajs[0] = start_points

    trajs = torch.tensor(trajs, dtype=torch.float32, device=DEVICE)
    video = video.to(DEVICE)
    with torch.no_grad():
        # inference
        for i in range(1, video_length):
            feat1 = model.extract_features(video[i-1].unsqueeze(0).to(DEVICE), trajs[i-1:i])
            next_pt = model.corr_softmax(feat1, video[i].unsqueeze(0), kps=trajs[i-1:i], search_size=5)
            trajs[i] = next_pt.squeeze(0)

    valids = np.ones((video_length, start_points.shape[0]))
    return trajs.cpu().numpy(), valids




