from yacs.config import CfgNode as CN
import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

_C = CN()
_C.PATH = CN()
_C.MODEL = CN()
_C.DATASETS = CN()
_C.SOLVER = CN()
_C.DATALOADER = CN()

_C.PATH.ROOT_PATH = '<path_project_dir>'
_C.PATH.INPUT_PATH = '<path_to_input_path>' # train or test

_C.MODEL.CUDA = 1 # gpu number
_C.MODEL.DEVICE = "<cuda or cpu>"
_C.MODEL.OUTPUT_DIR = "<save train weight path>"
_C.MODEL.NAME = "<model name>"
_C.MODEL.HIDDEN_DIM = 128 # hidden layer size

_C.DATASETS.NAME = "<dataset name>"

_C.DATALOADER.NUM_WORKERS = 8
_C.DATALOADER.NUM_INSTANCE = 4

_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 128

_C.SOLVER.OPTIMIZER_NAME = "Adam"

_C.SOLVER.MAX_EPOCHS = 50
_C.SOLVER.IMS_PER_BATCH = 128

_C.SOLVER.BASE_LR = 3e-4
_C.SOLVER.BIAS_LR_FACTOR = 2
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.MARGIN = 0.3

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0005

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (30, 55)

_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITERS = 500
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.CHECKPOINT_PERIOD = 50
_C.SOLVER.LOG_PERIOD = 100
_C.SOLVER.EVAL_PERIOD = 50


_C.merge_from_file(os.path.join(BASE_PATH, "config.yaml"))
_C.freeze()