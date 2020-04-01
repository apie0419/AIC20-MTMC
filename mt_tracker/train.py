import torch, os, logging
from config import cfg
from data import make_data_loader
from layers import make_loss
from solver import make_optimizer, WarmupMultiStepLR
from modeling import build_model
from utils.logger import setup_logger
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage

def create_trainer(model, optimizer, loss_fn, device=None):
    
    if device:
        model.to(device)

    def _update(engine, batch):
        model.train()
        
        data, target = batch

        data = data.cuda()
        target = target.cuda()
        output = model(data)
        loss = loss_fn(output, target)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        acc = (output.max(1)[1] == target).float().mean()

        return loss.item(), acc.item()

    return Engine(_update)

def create_evaluator(model, device=None):
    
    if device:
        model.to(device)

    def _update(engine, batch):
        model.eval()
        
        data, target = batch

        data = data.cuda()
        target = target.cuda()
        output = model(data)
        
        acc = (output.max(1)[1] == target).float().mean()

        return acc.item()

    return Engine(_update)

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
        torch.cuda.set_device(cfg.MODEL.CUDA)
        
    logger = logging.getLogger("tracker.train")
    logger.info("Start Training")
    trainer = create_trainer(model, optimizer, loss_fn, device=device)
    evaluator = create_evaluator(model, device=device)
    # checkpointer = ModelCheckpoint(dirname=output_dir, filename_prefix=cfg.MODEL.NAME, n_saved=None, require_empty=False)
    timer = Timer(average=True)

    # trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model,
    #                                                                  'optimizer': optimizer})
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')
    RunningAverage(output_transform=lambda x: x).attach(evaluator, 'eva_avg_acc')

    @trainer.on(Events.EPOCH_COMPLETED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        
        iter = (engine.state.iteration - 1) % len(train_loader) + 1
        if iter % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Accuracy: {:.3f},  Base Lr: {:.2e}"
                        .format(engine.state.epoch, iter, len(train_loader),
                                engine.state.metrics['avg_loss'],
                                engine.state.metrics['avg_acc'],
                                scheduler.get_lr()[0]))

    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            evaluator.run(val_loader)
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("Accuracy: {:.1%}".format(evaluator.state.metrics['eva_avg_acc']))
        if engine.state.epoch % checkpoint_period == 0:
            torch.save(model.state_dict(), os.path.join(output_dir, "model_" + cfg.MODEL.NAME + "_" + str(engine.state.epoch) + ".pth"))

    trainer.run(train_loader, max_epochs=epochs)

def main():
    output_dir = cfg.MODEL.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("tracker", output_dir, 0)
    logger.info("Running with config:\n{}".format(cfg))
    torch.backends.cudnn.benchmark = True

    train_loader, val_loader = make_data_loader(cfg)

    model = build_model(cfg)

    optimizer = make_optimizer(cfg, model)
    scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                  cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)

    loss_func = make_loss(cfg)

    do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_func
    )

if __name__ == '__main__':
    main()