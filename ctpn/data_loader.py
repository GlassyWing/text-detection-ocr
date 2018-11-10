import cv2
from glob import glob

from ctpn.lib.utils import random_uniform_num, readxml, cal_rpn, IMAGE_MEAN
import numpy as np
import os


class DataLoader:

    def __init__(self, anno_dir, images_dir):
        self.anno_dir = anno_dir
        self.images_dir = images_dir
        self.batch_size = 1

        # list xml
        self.xmlfiles = glob(f'{anno_dir}/*.xml')
        self.total_size = len(self.xmlfiles)
        self.__rd = random_uniform_num(self.total_size)
        self.xmlfiles = np.array(self.xmlfiles)
        self.steps_per_epoch = self.total_size // self.batch_size

    def load_data(self):
        xmlfiles = self.xmlfiles
        while True:
            shuf = xmlfiles[self.__rd.get(1)]
            gtbox, imgfile = readxml(shuf[0])
            img = cv2.imread(os.path.join(self.images_dir, imgfile))
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

            yield m_img, {'rpn_class_reshape': cls, 'rpn_regress_reshape': regr}
