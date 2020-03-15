# encoding: utf-8

import numpy as np
import pickle
import torch
from ignite.metrics import Metric

from .eval_dhn import *

class MOTA(Metric):
    def __init__(self):
        super(MOTA, self).__init__()

    def reset(self):
        self.matrices, self.targets = list(), list()
    

    def update(self, output):
        soft_assign_matrix, target = output
        self.matrices.append(soft_assign_matrix)
        self.targets.append(target)

    def compute(self):
        softmaxed_row = rowSoftMax(output_track_gt, scale=args.smax_scale).contiguous()
        softmaxed_col = colSoftMax(output_track_gt, scale=args.smax_scale).contiguous()

        
        fn = missedObjectPerframe(softmaxed_col)
        fp = falsePositivePerFrame(softmaxed_row)

        return cmc, mAP
