# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch


def train_collate_fn(batch):
    imgs, vids, camids = zip(*batch)
    vids = torch.tensor(vids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), vids, camids

def val_collate_fn(batch):
    imgs, vids, camids = zip(*batch)
    return torch.stack(imgs, dim=0), vids, camids

def test_collate_fn(batch):
    imgs, vids, camids = zip(*batch)
    return torch.stack(imgs, dim=0), vids, camids
