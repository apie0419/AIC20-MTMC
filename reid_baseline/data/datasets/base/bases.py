# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import numpy as np


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        vids, cams = list(), list()
        for _, vid, camid in data:
            vids += [vid]
            cams += [camid]
        vids = set(vids)
        cams = list(set(cams))
        num_vids = len(vids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_vids, num_imgs, num_cams

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_vids, num_train_imgs, num_train_cams = self.get_imagedata_info(train)
        num_query_vids, num_query_imgs, num_query_cams = self.get_imagedata_info(query)
        num_gallery_vids, num_gallery_imgs, num_gallery_cams = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_vids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_vids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_vids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")
