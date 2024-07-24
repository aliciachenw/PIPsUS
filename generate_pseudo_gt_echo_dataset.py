
from nets.pips2 import *
import saverloader
import torch
import csv
import os
import cv2
from ultrasound.pseudo_label_v2 import cvt_opencv_kps_to_numpy, extract_keypoints

new_path = 'D:/Wanwen/EchoNet/echonetdynamic-2/EchoNet-Dynamic/pseudo20'
ECHONET_PATH = 'D:/Wanwen/EchoNet/echonetdynamic-2/EchoNet-Dynamic/EchoNet-Dynamic/'

def read_video(video_path, shape):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, shape)
        frames.append(frame)
    cap.release()
    return np.stack(frames, axis=0)


step=seq_length=20

def main(reshape_size=(256, 256)):

    video_list = []
    # read csv
    with open(os.path.join(ECHONET_PATH,'FileList_mini.csv'), 'r') as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            video_list.append(row[0])
        
    device = 'cuda:0'
    # setup model and optimizer
    print("setting up model...")
    teacher_model = Pips(stride=8) #.to(device)
    _ = saverloader.load('./reference_model', teacher_model)
    teacher_model.eval()
    teacher_model.to(device)
    
    # load dataset
    print("loading data...")
    for i, video in enumerate(video_list):
        video_path = os.path.join(ECHONET_PATH, 'Videos', video + '.avi')
        print(video_path)
        rgbs = read_video(video_path, reshape_size) # S,H,W,C
        
        inference(rgbs, teacher_model, seq_length, step, device, video)


def inference(rgbs, model, S=16, step=10, device='cpu', filename=''):

    with torch.no_grad():
        video_length = rgbs.shape[0]

        for i in range(0, video_length-S, step):
            sub_video = rgbs[i:i+S]
            kps = extract_keypoints(sub_video[0], keypoint_type='sift', torch_tensor=False)
            if len(kps) == 0:
                print("len kps == 0")
                continue
            kps = cvt_opencv_kps_to_numpy(kps)

            # run inference
            trajs_0 = np.expand_dims(kps, axis=0).repeat(S, axis=0) # S x N x 2
            trajs_0 = np.expand_dims(trajs_0, axis=0)
            trajs_0 = torch.from_numpy(trajs_0).float().to(device)
            sub_video = torch.from_numpy(sub_video).permute(0,3,1,2).unsqueeze(0).float().to(device) # 1 x S x C x H x W
            preds, _, _, _ = model(trajs_0, sub_video.to(device), iters=16, feat_init=None, beautify=True)
            trajs_e = preds[-1].squeeze(0) # S x N x 2
            S, N, _ = trajs_e.shape
            trajs_e = trajs_e.reshape((S*N, 2))

            if filename:
                filename_i = filename + '_' + str(i).zfill(4) + '.csv'
                np.savetxt(os.path.join(new_path, filename_i), trajs_e.to('cpu').numpy())



if __name__ == '__main__':
    main()