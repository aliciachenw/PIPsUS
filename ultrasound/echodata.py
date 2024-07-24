import numpy as np
import cv2
import os
import torch
from torch.utils.data import Dataset
import csv

if os.name == 'nt':
    ECHONET_PATH = 'D:/Wanwen/EchoNet/echonetdynamic-2/EchoNet-Dynamic/EchoNet-Dynamic/'
else:
    ECHONET_PATH = '/workspace/us_seq_dataset/EchoNet/echonetdynamic-2/EchoNet-Dynamic/EchoNet-Dynamic/'
    
# train: 7465, val: 1288, test: 1277
class EchoUSDataset(Dataset):
    def __init__(self, split, shape, max_num=None, use_mini=False):

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

        self.len = len(self.videos)

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        rgbs = self.videos[idx]
        filename = self.video_list[idx]
        return {'rgbs': rgbs, 'filename': filename}
    
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