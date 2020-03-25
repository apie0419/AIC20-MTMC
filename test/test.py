import sys, os, torch
from PIL import Image

sys.path.append("..")

from config import cfg
from model import build_reid_model, build_mtmc_model, build_transforms

base_path = os.path.dirname(os.path.abspath(__file__))
query_image = os.path.join(base_path, "images/244/0000002706.jpg")
neg_image = os.path.join(base_path, "images/243/0000002650.jpg")
pos_image = os.path.join(base_path, "images/244/0000001111.jpg")

reid = build_reid_model(cfg)
mtmc = build_mtmc_model(cfg)
transform = build_transforms(cfg)

query_image = transform(Image.open(query_image).convert('RGB'))
neg_image = transform(Image.open(neg_image).convert('RGB'))
pos_image = transform(Image.open(pos_image).convert('RGB'))

reid = reid.to(cfg.DEVICE.TYPE)
mtmc = mtmc.to(cfg.DEVICE.TYPE)

with torch.no_grad():
    reid.eval()
    mtmc.eval()

    query_image = query_image.cuda()
    neg_image = neg_image.cuda()
    pos_image = pos_image.cuda()
    query_feature = list(reid(query_image.view(1, 3, 256, 256))[0])
    neg_feature = list(reid(neg_image.view(1, 3, 256, 256))[0])
    pos_feature = list(reid(pos_image.view(1, 3, 256, 256))[0])

    pos_feature = torch.FloatTensor(query_feature[:-1] + pos_feature[:-1]).view(1, cfg.MTMC.HIDDEN_DIM).cuda()
    neg_feature = torch.FloatTensor(query_feature[:-1] + neg_feature[:-1]).view(1, cfg.MTMC.HIDDEN_DIM).cuda()

    m = torch.nn.Softmax(dim=1)

    pos_prob = m(mtmc(pos_feature))[0]
    neg_prob = m(mtmc(neg_feature))[0]

    print (f"Positive Case: {pos_prob}")
    print (f"Negtive Case: {neg_prob}")

