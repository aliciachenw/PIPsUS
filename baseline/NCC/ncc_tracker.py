import numpy as np

from skimage.feature import match_template



def ncc_matching(image1, image2, kps, patch_size, search_size):
    # Fast Normalized Cross-Correlation
    kp_num = kps.shape[0]
    next_kps = np.zeros((kp_num, 2))
    # zero padding for the image
    pad_size = search_size + patch_size
    image1 = np.pad(image1, ((pad_size, pad_size), (pad_size, pad_size)), 'constant', constant_values=(0, 0))
    image2 = np.pad(image2, ((pad_size, pad_size), (pad_size, pad_size)), 'constant', constant_values=(0, 0))
    half_patch_size = patch_size // 2

    for i in range(kp_num):
        x = kps[i,0] + pad_size
        y = kps[i,1] + pad_size
        y = int(y)
        x = int(x)
        template_patch = image1[y-half_patch_size:y+half_patch_size+1, x-half_patch_size:x+half_patch_size+1]
        search_patch = image2[y-search_size-half_patch_size:y+search_size+half_patch_size+1, x-search_size-half_patch_size:x+search_size+half_patch_size+1]
        if template_patch.shape[0] == 0 or template_patch.shape[1] == 0 or search_patch.shape[0] == 0 or search_patch.shape[1] == 0:
            # print("Error: template patch size is 0")
            # print("x: %d, y: %d" %(x, y))
            # print(kps[i])
            # maybe the keypoint is out of the image, just skip
            next_kps[i] = kps[i]
            continue
        signal = match_template(search_patch, template_patch, pad_input=True)

        ij = np.unravel_index(np.argmax(signal), signal.shape)
        x_target, y_target = ij[::-1]
        center_x = search_size+half_patch_size
        center_y = search_size+half_patch_size
        next_kps[i,0] = x_target - center_x + x - pad_size
        next_kps[i,1] = y_target - center_y + y - pad_size

    return next_kps

# def ncc(patch1, patch2):
#     norm_patch1 = norm_data(patch1.flatten())
#     norm_patch2 = norm_data(patch2.flatten())
#     if norm_patch1.shape[0] != norm_patch2.shape[0]:
#         print("Error: patch size not match %d %d" %(norm_patch1.shape[0], norm_patch2.shape[0]))
#         return 0
#     result = np.dot(norm_patch1, norm_patch2) / norm_patch1.shape[0]
#     return result 


# def norm_data(data):
#     mean_data = np.mean(data)
#     std_data = np.std(data)
#     data = (data - mean_data) / (std_data + 1e-6)
#     # data = data / std_data
#     return data



# if __name__ == '__main__':
#     data = np.random.rand(100, 100)
#     flow = ncc_matching(data, data, np.array([[50,50]]), 8, 8)
#     print(flow)
#     print(ncc(data, data))