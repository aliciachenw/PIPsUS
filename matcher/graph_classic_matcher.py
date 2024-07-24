import pygmtools as pygm
import numpy as np
import torch
from scipy.spatial import distance_matrix


class MatchPair(object):
    def __init__(self):
        self.queryIdx = 0
        self.trainIdx = 0


class HungarianMatcher:
    def __init__(self):
        pass
    def match(self, descriptors1, descriptors2, backend):
        if backend == 'pytorch':
            if len(descriptors1.shape) == 3:
                descriptors1 = descriptors1.squeeze(0)
                descriptors2 = descriptors2.squeeze(0)
            return hungarian_solver(descriptors1, descriptors2, backend)
        else: # should be numpy
            if len(descriptors1.shape) == 3:
                descriptors1 = descriptors1[0]
                descriptors2 = descriptors2[0]
            return hungarian_solver(descriptors1, descriptors2, backend)

class SinkhornMatcher:
    def __init__(self):
        pass
    def match(self, descriptors1, descriptors2, backend):
        if backend == 'pytorch':
            if len(descriptors1.shape) == 3:
                descriptors1 = descriptors1.squeeze(0)
                descriptors2 = descriptors2.squeeze(0)
            return sinkhorn_solver(descriptors1, descriptors2, backend)
        else: # should be numpy
            if len(descriptors1.shape) == 3:
                descriptors1 = descriptors1[0]
                descriptors2 = descriptors2[0]
            return sinkhorn_solver(descriptors1, descriptors2, backend)


def hungarian_solver(descriptors1, descriptors2, backend):
    if backend == 'numpy':
        sim = descriptors1 @ descriptors2.T
    elif backend == 'pytorch':
        sim = descriptors1 @ descriptors2.t()
    x = pygm.hungarian(sim, backend=backend)

    # get matches
    matches = []
    for i in range(x.shape[0]):
        idx = np.where(x[i, :] == 1)[0]
        match = MatchPair()
        if len(idx) > 0:
            match.queryIdx = i
            match.trainIdx = idx[0]
            matches.append(match)
    # print("Find %d matches" % len(matches))
    return matches


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1

def sinkhorn_solver(descriptors1, descriptors2, backend):
    if backend != 'pytorch':
        raise NotImplementedError
    
    
    sim = descriptors1 @ descriptors2.t()
    x = pygm.sinkhorn(sim, tau=0.1, backend=backend).unsqueeze(0)
    if x.size(1) == 0 or x.size(2) == 0:
        return []
    # get matches
    max0, max1 = x.max(2), x.max(1)
    # print(max0, max1)
    indices0, indices1 = max0.indices, max1.indices
    # print(indices0, indices1)
    mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
    mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
    zero = x.new_tensor(0)
    mscores0 = torch.where(mutual0, max0.values.exp(), zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
    valid0 = mutual0
    valid1 = mutual1 & valid0.gather(1, indices1)
    indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
    indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))
    matches = []
    for i in range(indices0.shape[1]):
        if indices0[0, i] > -1:
            match = MatchPair()
            match.queryIdx = i
            match.trainIdx = indices0[0, i]
            matches.append(match)

    return matches
