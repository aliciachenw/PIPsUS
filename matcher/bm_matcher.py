import cv2
from utils.visualize_utils import *
from utils.keypoint_utils import *

def bm_matching(img1, img2, pred_dict1, pred_dict2, filename=None): 
    """ Brute force matching between two images
    """
    detector1 = pred_dict1['p1']
    detector2 = pred_dict2['p1']
    descriptor1 = pred_dict1['d1']
    descriptor2 = pred_dict2['d1']


    cv_kp1 = cvt_numpy_to_cv2_keypoint(detector1)
    cv_kp2 = cvt_numpy_to_cv2_keypoint(detector2)

    bfmatcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = bfmatcher.match(descriptor1, descriptor2)

    return cv_kp1, cv_kp2, matches





def brute_force_matching(img1, img2, pred_dict1, pred_dict2, knn=1, filename=None):  # bug: if crosscheck is true, knn must be 1
    """ Brute force matching between two images
    """
    # filter out keypoints with low score
    pred_dict1 = filter_keypoints(pred_dict1, 0.7)
    pred_dict2 = filter_keypoints(pred_dict2, 0.7)
    # only visualize top 100 keypoints
    pred_dict1 = top_k_keypoints(pred_dict1, 200)
    pred_dict2 = top_k_keypoints(pred_dict2, 200)


    detector1 = pred_dict1['p1']
    detector2 = pred_dict2['p1']
    descriptor1 = pred_dict1['d1']
    descriptor2 = pred_dict2['d1']


    cv_kp1 = cvt_numpy_to_cv2_keypoint(detector1)
    cv_kp2 = cvt_numpy_to_cv2_keypoint(detector2)
    img1 = img1 * 255
    img1 = img1.astype(np.uint8)
    img2 = img2 * 255
    img2 = img2.astype(np.uint8)

    bfmatcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bfmatcher.knnMatch(descriptor1, descriptor2, k=knn)

    matched_image = cv2.drawMatchesKnn(img1, cv_kp1, img2, cv_kp2, matches, None,
           matchColor=(0, 255, 0), matchesMask=None,
           singlePointColor=(255, 0, 0), flags=0)
    if filename:
        cv2.imwrite(filename, matched_image)
    else:
        cv2.imshow("matches", matched_image)
        cv2.waitKey(0)

    return matches, matched_image