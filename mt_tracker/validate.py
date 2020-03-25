import torch
from config import cfg
from data import make_train_loader, make_val_loader
from modeling import build_model
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage


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
    
def do_validate(cfg, model, val_loader):

    device = cfg.MODEL.DEVICE
    if device == "cuda":
        torch.cuda.set_device(cfg.MODEL.CUDA)
    evaluator = create_evaluator(model, device=device)
    RunningAverage(output_transform=lambda x: x).attach(evaluator, 'eva_avg_acc')
    evaluator.run(val_loader)

    print ("Accuracy: {:.1%}".format(evaluator.state.metrics['eva_avg_acc']))

def main():
    
    torch.backends.cudnn.benchmark = True

    val_loader = make_val_loader(cfg)

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