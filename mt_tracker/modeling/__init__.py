from .model import MCT

def build_model(cfg):
    model = MCT(cfg.MODEL.APPEARANCE_DIM, cfg.MODEL.PHYSIC_DIM)
    return model