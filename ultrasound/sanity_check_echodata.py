import SimpleITK as sitk
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch
import csv
# generate simple translation dataset for sanity check
from ultrasound.sanity_check_data_v2 import generate_random_sequence
SANITY_CHECK_FAKE_DATA_LENGTH = 40

# generate simple translation dataset for sanity check
def conv_smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def rand_trans_smooth(max_x, max_y, seq_length):
    # random dx
    dx = np.linspace(0, max_x, seq_length)

    m_type = np.random.randint(4)
    if m_type == 0:
        dx = dx
    elif m_type == 1:
        dx = np.concatenate((dx, dx[::-1][1:]))
    elif m_type == 2:
        dx = np.concatenate((dx, dx[::-1][1:], -dx[1:]))
    elif m_type == 3:
        dx = np.concatenate((dx, dx[::-1][1:], -dx[1:], -dx[::-1][1:]))
    
    # resample to the fixed seq length
    x = np.linspace(0, len(dx), seq_length)
    dx = np.interp(x, range(0, len(dx)), dx)


    dy = np.linspace(0, max_y, seq_length)
    m_type = np.random.randint(4)
    if m_type == 0:
        dy = dy
    elif m_type == 1:
        ddyx = np.concatenate((dy, dy[::-1][1:]))
    elif m_type == 2:
        dy = np.concatenate((dy, dy[::-1][1:], -dy[1:]))
    elif m_type == 3:
        dy = np.concatenate((dx, dx[::-1][1:], -dy[1:], -dy[::-1][1:]))
    
    # resample to the fixed seq length
    y = np.linspace(0, len(dy), seq_length)
    dy = np.interp(y, np.arange(0, len(dy)), dy)

    assert len(dx) == len(dy)
    return dx, dy


def rand_trans(max_trans, seq_length, smooth=False, window=10):
    dx = np.random.rand(seq_length)
    dy = np.random.rand(seq_length)
    # scale to -max_trans, +max_trans
    dx = (dx - 0.5) / 0.5 * max_trans
    dy = (dy - 0.5) / 0.5 * max_trans

    # conv filtering
    if smooth:
        dx = conv_smooth(dx, window)
        dy = conv_smooth(dy, window)
    return dx, dy

def generate_translation_sequence(img, seq_length=SANITY_CHECK_FAKE_DATA_LENGTH, smooth=False):
    # img: H,W,3
    seq = [img]
    
    # random dx
    # max_x = np.random.randint(-50, 51)
    # max_y = np.random.randint(-50, 51)
    # dx, dy = rand_trans_smooth(max_x, max_y, seq_length)

    max_x = 20 # 20 pixel motion??
    # dx, dy = rand_trans(max_x, seq_length, smooth=smooth)
    if smooth:
        dx, dy = rand_trans_smooth(max_x, max_x, seq_length)
    else:
        dx, dy = rand_trans(max_x, seq_length)

    for i in range(dx.shape[0]):
        new_image = translate_image(img, dx[i], dy[i])
        seq.append(new_image)
    seq = np.array(seq)
    seq = torch.from_numpy(seq)
    motion = np.vstack((dx, dy)).T # (seq_length, 2)
    motion = torch.from_numpy(motion)
    # print(seq.shape, motion.shape)
    return seq, motion


def translate_image(img, dx, dy):
    aug_mat = np.zeros((2, 3))
    aug_mat[0, 0] = 1
    aug_mat[1, 1] = 1
    aug_mat[0, 2] = dx
    aug_mat[1, 2] = dy
    h, w, c = img.shape
    aug_img = cv2.warpAffine(img, aug_mat, (w, h), flags=cv2.INTER_LINEAR)
    return aug_img

if os.name == 'nt':
    ECHONET_PATH = 'D:/Wanwen/EchoNet/echonetdynamic-2/EchoNet-Dynamic/EchoNet-Dynamic/'
else:
    ECHONET_PATH = '/workspace/us_seq_dataset/EchoNet/echonetdynamic-2/EchoNet-Dynamic/EchoNet-Dynamic/'
    
class EchoUSDataset(Dataset):
    def __init__(self, split, shape, max_num=None, smooth=False, randomseed=None, use_mini=False):

        self.video_list = []
        self.shape = shape
        self.videos = []

        # read csv
        if use_mini:
            with open(os.path.join(ECHONET_PATH,'FileList_mini.csv'), 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    if row[-1].lower() == split:
                        self.video_list.append(row[0])
        else:
            with open(os.path.join(ECHONET_PATH,'FileList.csv'), 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    if row[-1].lower() == split:
                        self.video_list.append(row[0])
        
        # read video
        if max_num is not None:
            self.video_list = self.video_list[:max_num]
        for i, video in enumerate(self.video_list):
            video_path = os.path.join(ECHONET_PATH, 'Videos', video + '.avi')
            rgbs = self.read_video(video_path, shape)
            self.videos.append(rgbs)
        if randomseed is not None:
            np.random.seed(randomseed)
        self.len = len(self.videos)
        self.smooth = smooth


    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        rgbs = self.videos[idx]

        rgbs, motion = generate_random_sequence(rgbs[0], smooth=self.smooth)  # NOTE: motion is the displacement relative to the first frame!
        filename = self.video_list[idx]
        return {'rgbs': rgbs, 'motion': motion, 'filename': filename} # (S, H, W, C)
    
    def read_video(self, video_path, shape):
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