# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch


def train_collate_fn(batch):
    distance_matrices, gt_matrices = zip(*batch)

    for d in distance_matrices:
        print (d.shape)
        print (type(d))
    distance_matrices = torch.stack(distance_matrices, dim=0)
    gt_matrices = torch.stack(gt_matrices, dim=0)
    print (distance_matrices)
    
    return distance_matrices, gt_matrices

def val_collate_fn(batch):
    imgs, vids, camids = zip(*batch)
    return torch.stack(imgs, dim=0), vids, camids


