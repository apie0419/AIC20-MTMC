from yacs.config import CfgNode as CN
import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

_C = CN()
_C.PATH = CN()
_C.DEVICE = CN()

_C.PATH.ROOT_PATH = '<path_project_dir>'
_C.PATH.INPUT_PATH = '<path_to_input_path>' # train or test
_C.PATH.RESNET_PRETRAIN_MODEL_PATH = "<path_to_resnet_pretrain_model>" # resnet50-19c8e357.pth
_C.PATH.REID_MODEL_PATH = "<path_to_reid_model>"
_C.DEVICE.GPU = 1 # gpu number
_C.DEVICE.TYPE = "<cuda or cpu>"


_C.merge_from_file(os.path.join(BASE_PATH, "config.yaml"))