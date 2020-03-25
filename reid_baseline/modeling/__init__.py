# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .baseline import Baseline
import torch

def build_model(cfg, num_classes):
    if cfg.MODEL.NAME == 'resnet50':
        model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH)
        weight = torch.load("/home/apie/projects/AIC20-MTMC/weights/resnet50_model_t2.pth")
        model.load_state_dict(weight)
    return model
