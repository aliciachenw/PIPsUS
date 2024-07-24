import SimpleITK as sitk
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

def read_mha(mha_filename, reshape_size=None):
    # Read the image using SimpleITK
    # reshape_size: (W, H) if need to resize, None if no need to resize
    image = sitk.ReadImage(mha_filename)
    image = sitk.GetArrayFromImage(image)
    image = image.astype(np.uint8)
    # print(image.shape)
    if reshape_size is not None: 
        resize_image = []
        for i in range(image.shape[0]):
            resize_image.append(cv2.resize(image[i], reshape_size, interpolation=cv2.INTER_LINEAR))
        image = np.stack(resize_image, axis=0)
        image = image.astype(np.uint8)
    return image


def cvt_grays_to_rgbs(grays):
    rgbs = []
    for i in range(grays.shape[0]):
        rgbs.append(cv2.cvtColor(grays[i], cv2.COLOR_GRAY2RGB))
    rgbs = np.stack(rgbs, axis=0)
    return rgbs


class USDataset(Dataset):
    def __init__(self, split, shape, S=40, step=None):
        # shape = (w, h)
        # root_dir = '/workspace/us_seq_dataset/2D_cleaned_v2'
        if os.name == 'nt':
            root_dir = 'D:/Wanwen/TORS/us_us_registration_dataset/2D_cleaned_v2'
        else:
            root_dir = '/workspace/us_seq_dataset/2D_cleaned_v2'
        if split == 'train':
            patient_list =  ["OR_01192023_case_1",  "OR_04202023_case6", "OR_05122023_case8",\
                        "OR_04202023_case7", "OR_05122023_case9", "OR_07212023_Surgery1", "OR_07212023_Surgery2",\
                            "OR_07212023_Surgery3", "OR_08032023_Surgery1", "OR_09072023_Surgery1",\
                            "OR_09072023_Surgery2", "OR_09072023_Surgery3"]
        elif split == 'valid':
            patient_list =  ["OR_01262023_case_3", "OR_02232023_case4", "OR_06152023_Surgery2", "OR_03152023_case5"]
        elif split == 'test':
            patient_list = ["OR_01192023_case_2", "OR_06152023_Surgery1", "OR_06152023_Surgery3"]

        sub_floders = ['1_BeforeRetraction', '2_AfterRetraction']
        # scanning_types = ['1_Neck', '2_SMG', '2_Below_Chin', '3_BOT']
        # scanning_types = ['1_Neck', '2_SMG', '2_Below_Chin']
        scanning_types = ['1_Neck']
        self.video_list = []
        self.shape = shape

        self.filepaths = []

        if step is None:
            step = S

        for patient in patient_list:
            for sub_floder in sub_floders:
                for scan in scanning_types:
                    video_path = os.path.join(root_dir, patient, sub_floder, scan)
                    if os.path.exists(video_path):
                        filenames = os.listdir(video_path)
                        filenames.sort()
                        for fn in filenames:
                            if fn.endswith('.igs.mha'):
                                # print('reading', os.path.join(video_path, fn))
                                video = read_mha(os.path.join(video_path, fn), reshape_size=shape)
                                video_length = video.shape[0]
                                if video_length > S:
                                    for i in range(0, video_length-S, step):
                                        self.video_list.append(video[i:i+S])
                                # self.filepaths.append(patient + '/' + sub_floder + '/' + scan + '/' + fn[:-8])
                                # self.filepaths.append(os.path.join(patient, sub_floder, scan, fn[:-8]))
                    #         break
                    # break
            #     break
            # break

        self.len = len(self.video_list)
        if split == 'train' or 'valid':
            self.is_train = True
        else:
            self.is_train = False


    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        rgbs = cvt_grays_to_rgbs(self.video_list[idx])
        return {'rgbs': rgbs} # (S, H, W, C)
    


# class ToyUSDataset(Dataset):
#     def __init__(self, shape):
#         # shape = (w, h)
#         if os.name == 'nt':
#             root_dir = 'D:/Wanwen/TORS/us_us_registration_dataset/inplane_motion_test_data'
#         else:
#             root_dir = '/workspace/us_seq_dataset/inplane_motion_test_data'

#         patient_list = ['1']

#         sub_floders = ['preop', 'after']
#         scanning_types = ['carotid', 'smg', 'tonguebase']

#         self.video_list = []
#         self.shape = shape

#         self.filepaths = []

#         for patient in patient_list:
#             for sub_floder in sub_floders:
#                 for scan in scanning_types:
#                     video_path = os.path.join(root_dir, patient, sub_floder, scan)
#                     if os.path.exists(video_path):
#                         filenames = os.listdir(video_path)
#                         filenames.sort()
#                         for fn in filenames:
#                             if fn.endswith('.mha'):
#                                 # print('reading', os.path.join(video_path, fn))
#                                 video = read_mha(os.path.join(video_path, fn), reshape_size=shape)
#                                 self.video_list.append(video)
#                                 self.filepaths.append(patient + '/' + sub_floder + '/' + scan + '/' + fn[:-3])
#                                 # self.filepaths.append(os.path.join(patient, sub_floder, scan, fn[:-3]))
#             #                 break
#             #         break
#             #     break
#             # break

#         self.len = len(self.video_list)


#     def __len__(self):
#         return self.len

#     def __getitem__(self, idx):
#         rgbs = cvt_grays_to_rgbs(self.video_list[idx])
#         filename = self.filepaths[idx]
#         return {'rgbs': rgbs, 'filename': filename} # (S, H, W, C)