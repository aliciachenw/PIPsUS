import os
import numpy as np
import SimpleITK as sitk
import cv2


def mha_to_mp4(mha_filename, mp4_filename):
    # Read the image using SimpleITK
    image = sitk.ReadImage(mha_filename)
    image = sitk.GetArrayFromImage(image)
    
    print("seq length:", image.shape[0])
    print("image size:", image.shape[1], image.shape[2])

    out = cv2.VideoWriter(mp4_filename, cv2.VideoWriter_fourcc(*'mp4v'), 30, (image.shape[2], image.shape[1]))
    for i in range(image.shape[0]):
        image_ = image[i]
        
        image_ = image_.astype(np.uint8)
        image_ = cv2.cvtColor(image_, cv2.COLOR_GRAY2RGB)
        out.write(image_)
        # if i > 40:
        #     break
    out.release()


def main():
    mha_filename = 'D:/Wanwen/TORS/us_us_registration_dataset/2D_cleaned_v2/OR_01192023_case_1/1_BeforeRetraction/1_Neck/case1_seq_081537_0006.igs.mha'
    mha_filename = 'D:/Wanwen/TORS/us_us_registration_dataset/2D_cleaned_v2/OR_01192023_case_1/1_BeforeRetraction/2_SMG/case1_seq_082042_0001.igs.mha'
    mp4_filename = 'case1_081537_0006.mp4'
    mp4_filename = 'case1_082042_0001.mp4'
    mha_to_mp4(mha_filename, mp4_filename)

if __name__ == '__main__':
    main()