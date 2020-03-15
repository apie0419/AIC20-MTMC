import logging

import torch, os
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage

def create_supervised_trainer(model, optimizer, loss_fn, device=None):
    
    if device:
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        data, target = batch
        data = data.cuda()
        target = target.cuda()
        assignment = model(data)
        loss = loss_fn(assignment, target)
        loss.backward()
        optimizer.step()
        
        return loss.item()

    return Engine(_update)


def create_supervised_evaluator(model, metrics, device=None):
    
    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, target = batch
            data = data.cuda()
            feat = model(data)
            return feat, target

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.MODEL.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS
    if device == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.MODEL.CUDA)
        torch.cuda.set_device(cfg.MODEL.CUDA)
        

    logger = logging.getLogger("dhn_baseline.train")
    logger.info("Start training")
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
    # evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query)}, device=device)
    checkpointer = ModelCheckpoint(dirname=output_dir, filename_prefix=cfg.MODEL.NAME, n_saved=None, require_empty=False)
    timer = Timer(average=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model,
                                                                     'optimizer': optimizer})
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1

        if iter % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, iter, len(train_loader),
                                engine.state.metrics['avg_loss'],
                                scheduler.get_lr()[0]))

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    # @trainer.on(Events.EPOCH_COMPLETED)
    # def log_validation_results(engine):
    #     if engine.state.epoch % eval_period == 0:
    #         # evaluator.run(val_loader)
    #         # cmc, mAP = evaluator.state.metrics['r1_mAP']
    #         logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
    #         logger.info("mAP: {:.1%}".format(mAP))
    #         for r in [1, 5, 10]:
    #             logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    trainer.run(train_loader, max_epochs=epochs)
