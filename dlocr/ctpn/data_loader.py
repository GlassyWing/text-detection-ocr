import os
from concurrent.futures import ThreadPoolExecutor
from glob import glob

import cv2
import numpy as np

from dlocr.ctpn.lib.utils import random_uniform_num, readxml, cal_rpn, IMAGE_MEAN


class DataLoader:

    def __init__(self, anno_dir, images_dir, cache_size=64):
        self.anno_dir = anno_dir
        self.images_dir = images_dir
        self.batch_size = 1

        # list xml
        self.xmlfiles = glob(anno_dir + '/*.xml')
        self.total_size = len(self.xmlfiles)
        self.cache_size = cache_size
        self.__rd = random_uniform_num(self.total_size)
        self.__data_queue = []
        self.xmlfiles = np.array(self.xmlfiles)
        self.steps_per_epoch = self.total_size // self.batch_size
        self.__init_queue()

    def __init_queue(self):
        with ThreadPoolExecutor() as executor:
            for data in executor.map(lambda xml_path: self.__single_sample(xml_path),
                                     self.xmlfiles[self.__rd.get(self.cache_size)]):
                self.__data_queue.append(data)

    def __single_sample(self, xml_path):
        gtbox, imgfile = readxml(xml_path)
        img = cv2.imread(os.path.join(self.images_dir, imgfile))
        return gtbox, imgfile, img

    def load_data(self):

        while True:

            if len(self.__data_queue) == 0:
                self.__init_queue()

            gtbox, imgfile, img = self.__data_queue.pop(0)
            h, w, c = img.shape

            # clip image
            if np.random.randint(0, 100) > 50:
                img = img[:, ::-1, :]
                newx1 = w - gtbox[:, 2] - 1
                newx2 = w - gtbox[:, 0] - 1
                gtbox[:, 0] = newx1
                gtbox[:, 2] = newx2

            [cls, regr], _ = cal_rpn((h, w), (int(h / 16), int(w / 16)), 16, gtbox)
            # zero-center by mean pixel
            m_img = img - IMAGE_MEAN
            m_img = np.expand_dims(m_img, axis=0)

            regr = np.hstack([cls.reshape(cls.shape[0], 1), regr])

            #
            cls = np.expand_dims(cls, axis=0)
            cls = np.expand_dims(cls, axis=1)
            # regr = np.expand_dims(regr,axis=1)
            regr = np.expand_dims(regr, axis=0)

            yield m_img, {'rpn_class': cls, 'rpn_regress': regr}
