import os
import pandas as pd
from sklearn.utils import shuffle
from sklearn import preprocessing
from .base import BaseImageDataset

class aic20_t3(BaseImageDataset):

    def __init__(self, **kwargs):
        super(aic20_t3, self).__init__()
        self.root      = "/home/apie/projects/AIC20-MTMC/dataset/AIC20_T3"
        self.train_dir = os.path.join(self.root, 'train')

        self.train, self.query, self.gallery = self._process_train(500)
        self.num_vids, self.num_imgs, self.num_cams = self.get_imagedata_info(self.train)
        
    def _process_train(self, num_query):
        df = pd.DataFrame()
        imgs, vids, cams = list(), list(), list()
        for scene_dir in os.listdir(self.train_dir):
            if not scene_dir.startswith("S0"):
                continue
            for camera_dir in os.listdir(os.path.join(self.train_dir, scene_dir)):
                if not camera_dir.startswith("c0"):
                    continue
                camid = int(camera_dir[1:])
                data_dir = os.path.join(self.train_dir, scene_dir, camera_dir, "reid_images")
                for vid in os.listdir(data_dir):
                    img_list = os.listdir(os.path.join(data_dir, vid))
                    vids.extend([vid] * len(img_list))
                    cams.extend([camid] * len(img_list))
                    imgs.extend([os.path.join(data_dir, vid, img) for img in img_list])
        
        df['img_path'] = imgs
        df['vid'] = vids
        df["cams"] = cams

        vid_le = preprocessing.LabelEncoder()
        cam_le = preprocessing.LabelEncoder()

        df['v'] = vid_le.fit_transform(df['vid'].tolist())
        df['c'] = cam_le.fit_transform(df['cams'].tolist())

        trn_df = df[df['v']<128]
        val_df = df[df['v']>=128]
        trn_set = trn_df[['img_path', 'v', 'c']].values.tolist()
        val_set = val_df[['img_path', 'v', 'c']].values.tolist()
        shuffle(val_set)
        q,g = val_set[:num_query], val_set[num_query:]
        return trn_set, q, g

