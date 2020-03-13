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
        self.test_dir  = os.path.join(self.root, "test")

        self.train, self.query, self.gallery = self._process_train(500)
        self.query_test, self.gallery_test = self._process_test()
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

    def _process_test(self):
        gallery_imgs = list()
        
        for scene_dir in os.listdir(self.test_dir):
            for camera_dir in os.listdir(os.path.join(self.test_dir, scene_dir)):
                data_dir = os.path.join(self.test_dir, scene_dir, camera_dir, "cropped_images")
                img_list = os.listdir(data_dir)
                gallery_imgs.extend([os.path.join(data_dir, img) for img in img_list])
    
        query_imgs = [gallery_imgs[0]]
        query = [[img, -1, -1] for img in query_imgs]
        gallery = [[img, -1, -1] for img in gallery_imgs]
        return query, gallery

if __name__ == '__main__':
    df = process_trn() 
    print(df.head())
