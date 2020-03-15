from .dhn import Munkrs

def build_model(cfg):
    is_train = True
    if cfg.MODEL.DEVICE == "cuda":
        is_cuda = True
    else:
        is_cuda = False
    model = Munkrs(cfg.MODEL.ELEMENT_DIM, cfg.MODEL.HIDDEN_DIM, cfg.MODEL.TARGET_SIZE, cfg.MODEL.BI_DIRECTION, cfg.SOLVER.IMS_PER_BATCH, is_cuda, is_train)
    return model