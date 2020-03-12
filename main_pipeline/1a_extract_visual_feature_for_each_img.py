import torch, sys, os, re
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from ignite.engine    import Engine
from PIL              import Image

sys.path.append("..")

from model.reid import Baseline
from config     import cfg

PRETRAIN_PATH = cfg.PATH.RESNET_PRETRAIN_MODEL_PATH
WEIGHT        = cfg.PATH.REID_MODEL_PATH
INPUT_DIR     = cfg.PATH.INPUT_PATH
DEVICE        = cfg.DEVICE.TYPE
if DEVICE == "cuda":
    torch.cuda.set_device(cfg.DEVICE.GPU)  

PIXEL_MEAN = [0.485, 0.456, 0.406]
PIXEL_STD  = [0.229, 0.224, 0.225]
SIZE_TRAIN = [256,256]
SIZE_TEST  = [256,256]
PROB          = 0.5
PADDING       = 10
LAST_STRIDE   = 1
NUM_WORKERS   = 8
IMS_PER_BATCH = 64

class RandomErasing(object):
    
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img

class ImageDataset(Dataset):

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def read_image(self, img_path):
        got_img = False
        if not os.path.exists(img_path):
            raise IOError("{} does not exist".format(img_path))
        while not got_img:
            try:
                img = Image.open(img_path).convert('RGB')
                got_img = True
            except IOError:
                print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
                exit()
        return img

    def __getitem__(self, index):
        img_path, pid, camids = self.dataset[index]
        img = self.read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camids, img_path

def collate_fn(batch):
    imgs, vids, camids, path = zip(*batch)
    return torch.stack(imgs, dim=0), vids, camids, path

def build_transforms():
    normalize_transform = T.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)
    
    transform = T.Compose([
        T.Resize(SIZE_TEST),
        T.ToTensor(),
        normalize_transform
    ])

    return transform

def build_model():
    weight = torch.load(WEIGHT)
    NUM_CLASSES = weight["classifier.weight"].shape[0]
    model = Baseline(NUM_CLASSES, LAST_STRIDE, PRETRAIN_PATH)
    model.load_state_dict(weight)
    return model

def _process_data():
    gallery_imgs = list()
    
    for scene_dir in os.listdir(INPUT_DIR):
        if not scene_dir.startswith("S0"):
            continue
        for camera_dir in os.listdir(os.path.join(INPUT_DIR, scene_dir)):
            for not camera_dir.startswith("c0"):
                continue
            data_dir = os.path.join(INPUT_DIR, scene_dir, camera_dir, "cropped_images")
            img_list = os.listdir(data_dir)
            gallery_imgs.extend([os.path.join(data_dir, img) for img in img_list])

    query_imgs = [gallery_imgs[0]]
    query = [[img, -1, -1] for img in query_imgs]
    gallery = [[img, -1, -1] for img in gallery_imgs[1:]]
    return query, gallery

def _inference(model, data_loader):

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, _, _, paths = batch
            data = data.cuda()
            feat = model(data)
            
            for i,p in enumerate(paths):
                scene_dir = re.search(r"S([0-9]){2}", p).group(0)
                camera_dir = re.search(r"c([0-9]){3}", p).group(0)
                path = os.path.join(INPUT_DIR, scene_dir, camera_dir)
                with open(os.path.join(path, 'deep_features.txt'), 'a+') as f:
                    line = p.split('/')[-1]+ ' '
                    print(line)
                    feature = list(feat[i].cpu().numpy())
                    for fea in feature:
                        line = line + str(fea)+' '
                    f.write(line.strip()+'\n')

            return feat, paths

    evaluator = Engine(_inference)
    evaluator.run(data_loader)

if __name__ == "__main__":
    query, gallery = _process_data()
    transforms = build_transforms()
    model = build_model()
    model = model.to(DEVICE)
    dataset = ImageDataset(query + gallery, transforms)
    dataloader = DataLoader(dataset, batch_size=IMS_PER_BATCH, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)
    _inference(model, dataloader)
