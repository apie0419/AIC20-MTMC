from yacs.config import CfgNode as CN
import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

_C = CN()
_C.PATH = CN()
_C.DEVICE = CN()
_C.REID   = CN()
_C.MTMC   = CN()

_C.PATH.ROOT_PATH = '<path_project_dir>'
_C.PATH.INPUT_PATH = '<path_to_input_path>' # train or test

# For Reid Initialization
_C.REID.RESNET_PRETRAIN_MODEL_PATH = "<path_to_resnet_pretrain_model>" # resnet50-19c8e357.pth
_C.REID.MODEL_PATH = "<path_to_reid_model>"
_C.REID.NAME = "<dataset name>"
_C.REID.LAST_STRIDE = 1

# For Reid Image Transform
_C.REID.PIXEL_MEAN = [0.485, 0.456, 0.406]
_C.REID.PIXEL_STD =  [0.229, 0.224, 0.225] 
_C.REID.SIZE_TEST = [256,256]

# For MTMC Initialization
_C.MTMC.MODEL_PATH = "<path_to_mtmc_model>"
_C.MTMC.HIDDEN_DIM = 4094


_C.DEVICE.GPU = 1 # gpu number
_C.DEVICE.TYPE = "<cuda or cpu>"


_C.merge_from_file(os.path.join(BASE_PATH, "config.yaml"))