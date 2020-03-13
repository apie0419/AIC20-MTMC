import torch, sys
import torchvision.transforms as T
from .reid import Baseline

sys.path.append("..")

from config import cfg

PRETRAIN_PATH = cfg.PATH.RESNET_PRETRAIN_MODEL_PATH
WEIGHT        = cfg.PATH.REID_MODEL_PATH

PIXEL_MEAN = [0.485, 0.456, 0.406]
PIXEL_STD  = [0.229, 0.224, 0.225]
SIZE_TEST  = [256,256]
LAST_STRIDE   = 1

def build_model():
    weight = torch.load(WEIGHT)
    NUM_CLASSES = weight["classifier.weight"].shape[0]
    model = Baseline(NUM_CLASSES, LAST_STRIDE, PRETRAIN_PATH)
    model.load_state_dict(weight)
    return model

def build_transforms():
    normalize_transform = T.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)
    
    transform = T.Compose([
        T.Resize(SIZE_TEST),
        T.ToTensor(),
        normalize_transform
    ])

    return transform

    