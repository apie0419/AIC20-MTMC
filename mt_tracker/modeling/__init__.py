from .model import MTC

def build_model(cfg):
    model = MTC(cfg.MODEL.HIDDEN_DIM)
    return model