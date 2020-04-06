from .model import MCT

def build_model(cfg):
    model = MCT(cfg.MODEL.HIDDEN_DIM)
    return model