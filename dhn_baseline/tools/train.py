import os, sys
from torch.backends import cudnn

sys.path.append('..')

from config import cfg
from data import make_data_loader
from engine.trainer import do_train
from modeling import build_model
from layers import make_loss
from solver import make_optimizer, WarmupMultiStepLR
from utils.logger import setup_logger

def train(cfg):
    print(cfg)
    # prepare dataset
    train_loader, val_loader = make_data_loader(cfg)
    # prepare model
    model = build_model(cfg)

    optimizer = make_optimizer(cfg, model)
    scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                  cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)

    loss_func = make_loss()

    arguments = {}

    do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_func
    )


def main():
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    cfg.freeze()

    output_dir = cfg.MODEL.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info("Running with config:\n{}".format(cfg))

    cudnn.benchmark = True
    train(cfg)


if __name__ == '__main__':
    main()
