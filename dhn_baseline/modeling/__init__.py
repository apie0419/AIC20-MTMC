from .dhn import Munkrs
import torch

def build_model(cfg):
    is_train = True
    if cfg.MODEL.DEVICE == "cuda":
        is_cuda = True
        gpu_id = cfg.MODEL.CUDA
    else:
        is_cuda = False
        gpu_id = -1
    model = Munkrs(cfg.MODEL.ELEMENT_DIM, cfg.MODEL.HIDDEN_DIM, cfg.MODEL.TARGET_SIZE, cfg.MODEL.BI_DIRECTION, cfg.SOLVER.IMS_PER_BATCH, is_cuda, gpu_id, is_train)
    # weight = torch.load(cfg.PATH.DHN_PRETRAIN_MODEL_PATH)
    # model_weight = model.state_dict()
    # model_weight.update(weight)
    # model.load_state_dict(model_weight)
   
    model.hidden_row = model.init_hidden(1)
    model.hidden_col = model.init_hidden(1)

    return model