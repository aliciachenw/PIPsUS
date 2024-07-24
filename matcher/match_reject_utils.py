import numpy as np

def reject_match(kp1, kp2, matches, threshold=10):
    # threshold in pixel
    new_matches = []
    for match in matches:
        query_idx = match.queryIdx
        train_idx = match.trainIdx
        pt1 = kp1[query_idx].pt
        pt2 = kp2[train_idx].pt
        dist = np.sqrt((pt2[0] - pt1[0])**2+(pt2[1] - pt1[1])**2)
        if dist < threshold:
            new_matches.append(match)
    return new_matches