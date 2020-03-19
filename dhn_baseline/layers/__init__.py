import torch
from .focal_loss import FocalLoss

def make_loss(cfg):
    focal = FocalLoss()
    if cfg.MODEL.DEVICE == "cuda":
        torch.cuda.set_device(cfg.MODEL.CUDA)
        focal = focal.cuda()
    
    def loss_func(dist, assign):
        return focal(dist, assign)
        
    return loss_func