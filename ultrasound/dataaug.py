import numpy as np
import cv2
import scipy.ndimage as ndimage
import torch

def adjust_intensity(img, max_gain, smooth_sigma, noise_sigma):
    device = img.device
    img = img.cpu().numpy()

    gain = np.random.randint(-max_gain, max_gain)
    img = img + gain
    
    # if np.random.rand() > 0.7:
    #     img = ndimage.gaussian_filter(img, sigma=(0, 0, smooth_sigma, smooth_sigma))

    if np.random.rand() > 0.5:
        gauss = np.random.normal(0, noise_sigma, img.shape)
        img = img + gauss

    img = np.where(img > 0, img, 0)
    img = np.where(img < 255, img, 255)
    img = img.astype(np.uint8)      
    img = torch.from_numpy(img).to(device)
    return img