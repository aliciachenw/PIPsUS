import time
import numpy as np
import saverloader
from nets.pips2 import Pips
import utils.improc
from utils.basic import print_, print_stats
import torch
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from fire import Fire
import sys
import cv2
from pathlib import Path

from ultrasound.pseudo_label import extract_keypoints, cvt_opencv_kps_to_numpy


def read_mp4(fn):
    vidcap = cv2.VideoCapture(fn)
    frames = []
    while(vidcap.isOpened()):
        ret, frame = vidcap.read()
        if ret == False:
            break
        frames.append(frame)
    vidcap.release()
    return frames

def run_model(model, rgbs, S_max=128, N=64, iters=16, sw=None, counter=0):
    rgbs = rgbs.cuda().float() # B, S, C, H, W

    B, S, C, H, W = rgbs.shape
    assert(B==1)

    # # pick N points to track; we'll use a uniform grid
    # N_ = np.sqrt(N).round().astype(np.int32)
    # grid_y, grid_x = utils.basic.meshgrid2d(B, N_, N_, stack=False, norm=False, device='cuda')
    # grid_y = 8 + grid_y.reshape(B, -1)/float(N_-1) * (H-16)
    # grid_x = 8 + grid_x.reshape(B, -1)/float(N_-1) * (W-16)
    # xy0 = torch.stack([grid_x, grid_y], dim=-1) # B, N_*N_, 2

    # get init keypoints
    kps = extract_keypoints(rgbs[0,0].permute(1,2,0), keypoint_type='sift')
    xy0 = cvt_opencv_kps_to_numpy(kps) # N x 2
    xy0 = torch.from_numpy(xy0).float().cuda().unsqueeze(0) # 1 x N x 2

    # zero-vel init
    trajs_e = xy0.unsqueeze(1).repeat(1,S,1,1)

    iter_start_time = time.time()
    
    preds, preds_anim, _, _ = model(trajs_e, rgbs, iters=iters, feat_init=None, beautify=True)
    trajs_e = preds[-1]

    iter_time = time.time()-iter_start_time
    print('inference time: %.2f seconds (%.1f fps)' % (iter_time, S/iter_time))

    if sw is not None and sw.save_this:
        rgbs_prep = utils.improc.preprocess_color(rgbs)
        sw.summ_traj2ds_on_rgbs('outputs/trajs_on_rgbs', trajs_e[0:1], utils.improc.preprocess_color(rgbs[0:1]), cmap='hot', linewidth=1, show_dots=False)
        rgb_save = sw.summ_traj2ds_on_rgbs('outputs/trajs_on_rgbs', trajs_e[0:1], utils.improc.preprocess_color(rgbs[0:1]), cmap='hot', linewidth=1, show_dots=False, only_return=True)
        # print('rgb_save', rgb_save.shape)
        # print(rgb_save.dtype)
        rgb_save = rgb_save[0]
        rgb_save = rgb_save.permute(0,2,3,1)
        # save the video
        out = cv2.VideoWriter('inplane_iter6_sift_1s_vis_' + str(counter).zfill(3) + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 4, (W, H))
        for i in range(rgb_save.shape[0]):
            out.write(cv2.cvtColor(rgb_save[i].numpy().astype(np.uint8), cv2.COLOR_RGB2BGR))
        out.release()
    return trajs_e


def main(
        filename='./stock_videos/camel.mp4',
        S=48, # seqlen
        N=1024, # number of points per clip
        stride=8, # spatial stride of the model
        timestride=1, # temporal stride of the model
        iters=16, # inference steps of the model
        image_size=(512,896), # input resolution
        max_iters=4, # number of clips to run
        shuffle=False, # dataset shuffling
        log_freq=1, # how often to make image summaries
        log_dir='./logs_demo',
        init_dir='./reference_model',
        device_ids=[0],
):

    # the idea in this file is to run the model on a demo video,
    # and return some visualizations
    
    exp_name = 'de00' # copy from dev repo

    print('filename', filename)
    name = Path(filename).stem
    print('name', name)
    
    rgbs = read_mp4(filename)
    rgbs = np.stack(rgbs, axis=0) # S,H,W,3
    rgbs = rgbs[:,:,:,::-1].copy() # BGR->RGB
    rgbs = rgbs[::timestride]
    S_here,H,W,C = rgbs.shape
    print('rgbs', rgbs.shape)

    # autogen a name
    model_name = "%s_%d_%d_%s" % (name, S, N, exp_name)
    import datetime
    model_date = datetime.datetime.now().strftime('%H:%M:%S')
    model_name = model_name # + '_' + model_date ## this will cause OS error in windows
    print('model_name', model_name)
    
    log_dir = 'logs_demo'
    writer_t = SummaryWriter(log_dir + '/' + model_name + '/t', max_queue=10, flush_secs=60)

    global_step = 0

    model = Pips(stride=8).cuda()
    parameters = list(model.parameters())
    if init_dir:
        _ = saverloader.load(init_dir, model)
    global_step = 0
    model.eval()

    idx = list(range(0, max(S_here-S,1), S))
    if max_iters:
        idx = idx[:max_iters]
    
    for si in idx:
        global_step += 1
        
        iter_start_time = time.time()

        sw_t = utils.improc.Summ_writer(
            writer=writer_t,
            global_step=global_step,
            log_freq=log_freq,
            fps=16,
            scalar_freq=int(log_freq/2),
            just_gif=True)

        rgb_seq = rgbs[si:si+S]
        rgb_seq = torch.from_numpy(rgb_seq).permute(0,3,1,2).to(torch.float32) # S,3,H,W
        rgb_seq = F.interpolate(rgb_seq, image_size, mode='bilinear').unsqueeze(0) # 1,S,3,H,W
        
        with torch.no_grad():
            trajs_e = run_model(model, rgb_seq, S_max=S, N=N, iters=iters, sw=sw_t, counter=si)

        iter_time = time.time()-iter_start_time
        
        print('%s; step %06d/%d; itime %.2f' % (
            model_name, global_step, max_iters, iter_time))
        


    writer_t.close()

if __name__ == '__main__':
    # Fire(main)
    main(filename='case1_082042_0001.mp4', S=5, image_size=(727, 698), iters=6)

    # inference time:
    # ...found checkpoint ./reference_model\model-000200000.pth
    # inference time: 8.90 seconds (4.5 fps)
    # track_40_1024_de00; step 000001/4; itime 209.44

    # inplane
    # inference time: 8.33 seconds (4.8 fps)
    # inplane_124611_40_1024_de00; step 000001/4; itime 199.35
    # inference time: 6.84 seconds (5.9 fps)
    # inplane_124611_40_1024_de00; step 000002/4; itime 191.09
    # inference time: 6.84 seconds (5.8 fps)
    # inplane_124611_40_1024_de00; step 000003/4; itime 190.66