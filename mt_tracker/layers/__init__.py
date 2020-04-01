import torch
import torch.nn.functional as F
from .focal_loss import FocalLoss

def make_loss(cfg):
    focal = FocalLoss()
    if cfg.MODEL.DEVICE == "cuda":
        torch.cuda.set_device(cfg.MODEL.CUDA)
        focal = focal.cuda()
    
    def loss_func(dist, target):
        # return F.cross_entropy(dist, target)
        return focal(dist, target)
        
    return loss_func