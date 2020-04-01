import torch
import torchvision.transforms as T
from .reid import Baseline
from .mtmc import MCT


def build_reid_model(cfg):
    PRETRAIN_PATH = cfg.REID.RESNET_PRETRAIN_MODEL_PATH
    WEIGHT        = cfg.REID.MODEL_PATH
    weight = torch.load(WEIGHT)
    NUM_CLASSES = weight["classifier.weight"].shape[0]
    model = Baseline(NUM_CLASSES, cfg.REID.LAST_STRIDE, PRETRAIN_PATH)
    model.load_state_dict(weight)
    return model

def build_mtmc_model(cfg):
    WEIGHT = cfg.MTMC.MODEL_PATH
    weight = torch.load(WEIGHT)
    model = MCT(cfg.MTMC.APPEARANCE_DIM, cfg.MTMC.PHYSIC_DIM)
    model.load_state_dict(weight)
    return model

def build_transforms(cfg):
    normalize_transform = T.Normalize(mean=cfg.REID.PIXEL_MEAN, std=cfg.REID.PIXEL_STD)
    
    transform = T.Compose([
        T.Resize(cfg.REID.SIZE_TEST),
        T.ToTensor(),
        normalize_transform
    ])

    return transform