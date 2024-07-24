import numpy as np
import torch
import torch.nn.functional as F
import cv2
DEVICE = 'cuda:0'


def flow_tracking(video, start_points):
    # using raft optical flow to track keypoints
    video_length, C, H, W = video.shape
    flows = np.zeros((video_length, H, W, 2))


    for i in range(1, video_length):
        image1 = video[i-1].permute(1,2,0).cpu().numpy()[:,:,0]
        image2 = video[i].permute(1,2,0).cpu().numpy()[:,:,0]
        # cvt to gray
        image1 = image1.astype(np.uint8)
        image2 = image2.astype(np.uint8)

        flow = cv2.calcOpticalFlowFarneback(image1, image2, None, pyr_scale=0.5, levels=4, winsize=11, iterations=3, poly_n=5, poly_sigma=1.1, flags=0) 
        flows[i] = flow

    trajs = np.zeros((video_length, start_points.shape[0], 2))
    trajs[0] = start_points

    for i in range(1, video_length):
        # wrapping the kp to the next frame based on optical flow
        trajs[i] = add_flow_points(flows[i], trajs[i-1], interpolate=True)
    valids = np.ones((video_length, start_points.shape[0]))

    return trajs, valids




#### flowvid
# https://github.com/diegoroyo/flowvid/blob/master/flowvid/core/util/add_flow.py
def _interpolate_flow(flow, fx, fy):
    px, py = int(fx), int(fy)
    cx, cy = px + 0.5, py + 0.5
    # get (x1, y1), (x2, y2) bounding square
    if fy - py > 0.5:
        y1 = cy
    else:
        y1 = cy - 1.0
    if fx - px > 0.5:
        x1 = cx
    else:
        x1 = cx - 1.0
    x2 = x1 + 1.0
    y2 = y1 + 1.0
    # bilinear interpolation
    t1 = flow[int(y1), int(x1), :] * (x2 - fx) * (y2 - fy)
    t2 = flow[int(y2), int(x1), :] * (x2 - fx) * (fy - y1)
    t3 = flow[int(y1), int(x2), :] * (fx - x1) * (y2 - fy)
    t4 = flow[int(y2), int(x2), :] * (fx - x1) * (fy - y1)
    return t1 + t2 + t3 + t4


def add_flows(flow1, flow2, interpolate):
    """
        Calculate the result of accumulating flow1 and flow2.
        CAUTION: assumes that both flow1 and flow2 have the same shape.
        :param interpolate: Use 4 closest pixels instead of just the closest one.
    """
    [h, w] = flow1.shape[0:2]

    indexes = np.empty((h, w, 2))
    x_values = np.repeat(np.reshape(np.arange(w), (1, w)), h, axis=0)
    y_values = np.repeat(np.reshape(np.arange(h), (h, 1)), w, axis=1)
    indexes[:, :, 0] = x_values
    indexes[:, :, 1] = y_values
    indexes = indexes + 0.5 + flow1

    if interpolate:
        pad = 1  # ignore outermost pixel row to account for interpolation bounds
        add_func = np.vectorize(
            _interpolate_flow, signature='(m,n,2),(),()->(2)')
    else:
        indexes = indexes.astype(np.int32)
        pad = 0  # don't need to ignore outer row bc no out-of-bounds problems
        add_func = np.vectorize(
            lambda flow, x, y: flow[y, x, :], signature='(m,n,2),(),()->(2)')

    x_points = indexes[:, :, 0]
    y_points = indexes[:, :, 1]
    x_points[x_points < pad] = pad
    x_points[x_points > w - 1 - pad] = w - 1 - pad
    y_points[y_points < pad] = pad
    y_points[y_points > h - 1 - pad] = h - 1 - pad

    return flow1 + add_func(flow2, x_points, y_points)


def add_flow_points(flow, points, interpolate: bool):
    """
        :param flow: [h, w, 2] (u, v components)
        :param points: [n, 2] ndarray (x0 y0)
        :param interpolate: Use 4 closest points to interpolate flow / use closest
        :returns: [n, 2] ndarray with the moved points
                    where (x, y) += flow[x, y]
    """
    if interpolate:
        new_points = np.copy(points)
        pad = 1  # ignore outermost pixel row to account for interpolation bounds

        add_func = np.vectorize(
            _interpolate_flow, signature='(m,n,2),(),()->(2)')
    else:
        new_points = points.astype(np.int32)
        pad = 0  # don't need to ignore outer row bc no out-of-bounds problems

        add_func = np.vectorize(
            lambda flow, x, y: flow[y, x, :], signature='(m,n,2),(),()->(2)')

    x_points = new_points[:, 0]
    y_points = new_points[:, 1]
    [h, w] = flow.shape[0:2]

    x_points[x_points < pad] = pad
    x_points[x_points > w - 1 - pad] = w - 1 - pad
    y_points[y_points < pad] = pad
    y_points[y_points > h - 1 - pad] = h - 1 - pad

    return points + add_func(flow, x_points, y_points)

