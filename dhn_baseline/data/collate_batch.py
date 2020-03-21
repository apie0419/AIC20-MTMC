import numpy as np
import torch


def collate_fn(batch):
    dis_mat, gt, size = zip(*batch)
    dis_mat = list(dis_mat)
    gt = list(gt)
    size = np.array(size)
    _max = np.max(size)

    for i in range(size.shape[0]):
        padding = _max - size[i][0]
        m = torch.nn.ZeroPad2d((0, padding, 0, padding))
        dis_mat[i] = m(dis_mat[i])
        pad = torch.zeros((_max ** 2, ))
        pad[:size[i][0]**2] = gt[i]
        gt[i] = pad
    
    dis_mat = torch.stack(dis_mat, dim=0)
    gt = torch.stack(gt, dim=0)
    return dis_mat, gt, size
