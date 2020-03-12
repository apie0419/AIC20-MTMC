import os
import pandas as pd
import xml.etree.ElementTree as ET
from sklearn.utils import shuffle
from sklearn import preprocessing
from .base import BaseImageDataset

class veri(BaseImageDataset):

    def __init__(self, **kwargs):
        super(veri, self).__init__()
        self.root      = "/home/apie/projects/AIC20-MTMC/dataset/VERI"
        self.train_dir = os.path.join(self.root, 'image_train')
        self.test_dir  = os.path.join(self.root, "image_test")
        self.query_dir = os.path.join(self.root, "image_query")

        self.train, self.query, self.gallery = self._process_train(500)
        # self.query_test, self.gallery_test = self._process_test()
        self.num_vids, self.num_imgs, self.num_cams = self.get_imagedata_info(self.train)
        
    def _process_train(self, num_query):
        df = pd.DataFrame()
        tree = ET.parse(os.path.join(self.root, 'train_label.xml'), parser=ET.XMLParser(encoding="utf-8")).getroot()
        imgs, vids, cams = list(), list(), list()
        for element in tree.iter(tag="Item"):
            attr = element.attrib
            imgs.append(os.path.join(self.train_dir ,attr["imageName"]))
            vids.append(int(attr["vehicleID"]))
            cams.append(int(attr["cameraID"][1:]))

        df['img_path'] = imgs
        df['vid'] = vids
        df["cams"] = cams

        vid_le = preprocessing.LabelEncoder()
        cam_le = preprocessing.LabelEncoder()

        df['v'] = vid_le.fit_transform(df['vid'].tolist())
        df['c'] = cam_le.fit_transform(df['cams'].tolist())

        trn_df = df[df['v']<500]
        val_df = df[df['v']>=500]
        trn_set = trn_df[['img_path', 'v', 'c']].values.tolist()
        val_set = val_df[['img_path', 'v', 'c']].values.tolist()
        shuffle(val_set)
        q,g = val_set[:num_query], val_set[num_query:]
        return trn_set, q, g

    def _process_test(self):
        pass

if __name__ == '__main__':
    dataset = veri()
    dataset.print_dataset_statistics(dataset.train, dataset.query, dataset.gallery)
    
