import torch
import numpy as np
from config import cfg
from data import make_data_loader
from modeling import build_model
from ignite.engine import Engine, Events
from ignite.handlers import Timer
from ignite.metrics import RunningAverage


def create_evaluator(model, device=None):
    
    if device:
        model.to(device)

    def _update(engine, batch):
        model.eval()
        
        with torch.no_grad():
            data, target = batch

            data = data.cuda()
            target = target.cuda()
            output = model(data)
            
            acc = (output.max(1)[1] == target).float().mean()

        return acc.item()

    return Engine(_update)
    
def do_validate(cfg, model, val_loader):

    device = cfg.MODEL.DEVICE
    if device == "cuda":
        torch.cuda.set_device(cfg.MODEL.CUDA)
    evaluator = create_evaluator(model, device=device)
    RunningAverage(output_transform=lambda x: x).attach(evaluator, 'eva_avg_acc')

    timer = Timer(average=True)

    timer.attach(evaluator, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    acc_list = list()

    @evaluator.on(Events.ITERATION_COMPLETED)
    def log_accuracy(engine):
        iter = (engine.state.iteration - 1) % len(val_loader) + 1
        print ("Iteration[{}/{}]".format(iter, len(val_loader)))
        acc_list.append(engine.state.metrics['eva_avg_acc'])
    
    evaluator.run(val_loader)
    print ("Validation Accuracy: {:1%}".format(np.array(acc_list).mean()))

def main():
    
    torch.backends.cudnn.benchmark = True

    train_loader, val_loader = make_data_loader(cfg)

    model = build_model(cfg)
    weight = torch.load(cfg.MODEL.TEST_MODEL)
    model.load_state_dict(weight)
    do_validate(
        cfg,
        model,
        val_loader
    )

if __name__ == '__main__':
    main()