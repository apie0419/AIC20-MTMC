# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import logging

import torch
from ignite.engine import Engine

from utils.reid_metric import R1_mAP,Distmat

def create_test_inference(model, num_query, device=None):
    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids, paths = batch
            data = data.cuda()
            feat = model(data)
            f = open('feature.txt', 'a')
            for i,p in enumerate(paths):
                line = p.split('/')[-1]+ ' '
                print(line)
                feature = list(feat[i].cpu().numpy())
                for fea in feature:
                    line = line + str(fea)+' '
                f.write(line.strip()+'\n')
            f.close()
            return feat, pids, camids, paths

    engine = Engine(_inference)

    metrics={'Distmat': Distmat(num_query)}
    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine

def create_supervised_evaluator(model, metrics,
                                device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids = batch
            data = data.cuda()
            feat = model(data)
            return feat, pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def inference(
        cfg,
        model,
        val_loader,
        num_query
):
    device = cfg.MODEL.DEVICE

    logger = logging.getLogger("reid_baseline.inference")
    logger.info("Start inferencing")
    evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query)},
                                            device=device)

    evaluator.run(val_loader)
    cmc, mAP = evaluator.state.metrics['r1_mAP']
    logger.info('Validation Results')
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))



def test_inference(
        cfg,
        model,
        test_loader,
        num_query
):
    device = cfg.MODEL.DEVICE
    torch.cuda.set_device(1)
    logger = logging.getLogger("reid_baseline.inference")
    logger.info("Start inferencing")

    evaluator = create_test_inference(model, num_query,  device=device)
    evaluator.run(test_loader)
