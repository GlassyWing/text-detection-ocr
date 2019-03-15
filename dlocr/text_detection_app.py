import os
from concurrent.futures import ThreadPoolExecutor
from math import *
from multiprocessing import Lock
from dlocr.ctpn import default_ctpn_weight_path, default_ctpn_config_path
from dlocr.densenet import default_densenet_weight_path, default_densenet_config_path, default_dict_path

import cv2
import numpy as np
from PIL import Image

from dlocr.ctpn import CTPN
from dlocr.densenet import DenseNetOCR
from dlocr.densenet import load_dict


def dumpRotateImage(img, degree, pt1, pt2, pt3, pt4):
    height, width = img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    matRotation = cv2.getRotationMatrix2D((width // 2, height // 2), degree, 1)
    matRotation[0, 2] += (widthNew - width) // 2
    matRotation[1, 2] += (heightNew - height) // 2
    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    pt1 = list(pt1)
    pt3 = list(pt3)

    [[pt1[0]], [pt1[1]]] = np.dot(matRotation, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(matRotation, np.array([[pt3[0]], [pt3[1]], [1]]))
    ydim, xdim = imgRotation.shape[:2]
    imgOut = imgRotation[max(1, int(pt1[1])): min(ydim - 1, int(pt3[1])),
             max(1, int(pt1[0])): min(xdim - 1, int(pt3[0]))]

    return imgOut


def sort_box(box):
    """
    对box进行排序
    """
    box = sorted(box, key=lambda x: sum([x[1], x[3], x[5], x[7]]))
    return box


def single_text_detect(rec, ocr, id_to_char, img, adjust):
    xDim, yDim = img.shape[1], img.shape[0]
    xlength = int((rec[2] - rec[0]) * 0.1)
    ylength = int((rec[3] - rec[1]) * 0.2)
    if adjust:
        pt1 = (max(1, rec[0] - xlength), max(1, rec[1] - ylength))
        pt2 = (rec[2], rec[3])
        pt3 = (min(rec[6] + xlength, xDim - 2), min(yDim - 2, rec[7] + ylength))
        pt4 = (rec[4], rec[5])
    else:
        pt1 = (max(1, rec[0]), max(1, rec[1]))
        pt2 = (rec[2], rec[3])
        pt3 = (min(rec[6], xDim - 2), min(yDim - 2, rec[7]))
        pt4 = (rec[4], rec[5])

    degree = degrees(atan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))  # 图像倾斜角度

    partImg = dumpRotateImage(img, degree, pt1, pt2, pt3, pt4)

    if partImg.shape[0] < 1 or partImg.shape[1] < 1 or partImg.shape[0] > partImg.shape[1]:  # 过滤异常图片
        return None

    image = Image.fromarray(partImg).convert('L')
    text, _ = ocr.predict(image, id_to_char)
    return image, text


def clip_single_img(bbox, img, xDim, yDim, adjust):
    xlength = int((bbox[2] - bbox[0]) * 0.1)
    ylength = int((bbox[3] - bbox[1]) * 0.2)
    if adjust:
        pt1 = (max(1, bbox[0] - xlength), max(1, bbox[1] - ylength))
        pt2 = (bbox[2], bbox[3])
        pt3 = (min(bbox[6] + xlength, xDim - 2), min(yDim - 2, bbox[7] + ylength))
        pt4 = (bbox[4], bbox[5])
    else:
        pt1 = (max(1, bbox[0]), max(1, bbox[1]))
        pt2 = (bbox[2], bbox[3])
        pt3 = (min(bbox[6], xDim - 2), min(yDim - 2, bbox[7]))
        pt4 = (bbox[4], bbox[5])

    degree = degrees(atan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))  # 图像倾斜角度

    partImg = dumpRotateImage(img, degree, pt1, pt2, pt3, pt4)
    image = Image.fromarray(partImg)
    return image


def clip_imgs_with_bboxes(bboxes, img, adjust):
    xDim, yDim = img.shape[1], img.shape[0]

    imgs = []
    with ThreadPoolExecutor() as executor:
        for img in executor.map(lambda t: clip_single_img(t[0], t[1], xDim, yDim, adjust),
                                map(lambda bbox: (bbox, img), bboxes)):
            imgs.append(img)
    return imgs


class TextDetectionApp:
    __lock = Lock()
    __ocr = None

    def __init__(self,
                 ctpn_weight_path,
                 densenet_weight_path,
                 dict_path,
                 ctpn_config_path=None,
                 densenet_config_path=None):
        """

        :param ctpn_weight_path:    CTPN 模型权重文件路径
        :param densenet_weight_path: Densenet 模型权重文件路径
        :param dict_path:           字典文件路径
        :param ctpn_config_path:    CTPN 模型配置文件路径
        :param densenet_config_path: Densenet 模型配置文件路径
        """

        self.id_to_char = load_dict(dict_path, encoding="utf-8")

        # 初始化CTPN模型
        if ctpn_config_path is not None:
            ctpn_config = CTPN.load_config(ctpn_config_path)
            ctpn_config["weight_path"] = ctpn_weight_path
            self.ctpn = CTPN(**ctpn_config)
        else:
            self.ctpn = CTPN()

        # 初始化Densenet 模型
        if densenet_config_path is not None:
            densenet_config = DenseNetOCR.load_config(densenet_config_path)
            densenet_config["weight_path"] = densenet_weight_path
            self.ocr = DenseNetOCR(**densenet_config)
        else:
            self.ocr = DenseNetOCR(num_classes=len(self.id_to_char))

    def detect(self, image, adjust=True, parallel=True):
        """

        :param parallel: 是否并行处理
        :param image: numpy数组形状为(h, w, c)或图像路径
        :param adjust: 是否调整检测框
        :return:
        """

        if type(image) == str:
            if not os.path.exists(image):
                raise ValueError("The image path: " + image + " not exists!")
        text_recs, img = self.ctpn.predict(image, mode=2)  # 得到所有的检测框

        if len(text_recs) == 0:
            return [], []

        text_recs = sort_box(text_recs)

        if parallel:
            imgs = clip_imgs_with_bboxes(text_recs, img, adjust)

            texts = self.ocr.predict_multi(imgs, id_to_char=self.id_to_char)
        else:
            texts = []
            for index, rec in enumerate(text_recs):
                image, text = single_text_detect(rec, self.ocr, self.id_to_char, img, adjust)  # 识别文字
                # plt.subplot(len(text_recs), 1, index + 1)
                # plt.imshow(image)
                if text is not None and len(text) > 0:
                    texts.append(text)

        return text_recs, texts

    @staticmethod
    def get_or_create(ctpn_weight_path=default_ctpn_weight_path,
                      ctpn_config_path=default_ctpn_config_path,
                      densenet_weight_path=default_densenet_weight_path,
                      densenet_config_path=default_densenet_config_path,
                      dict_path=default_dict_path):

        TextDetectionApp.__lock.acquire()
        try:
            if TextDetectionApp.__ocr is None:
                TextDetectionApp.__ocr = TextDetectionApp(ctpn_weight_path=ctpn_weight_path,
                                                          ctpn_config_path=ctpn_config_path,
                                                          densenet_weight_path=densenet_weight_path,
                                                          densenet_config_path=densenet_config_path,
                                                          dict_path=dict_path)
        except Exception as e:
            print(e)
        finally:
            TextDetectionApp.__lock.release()
        return TextDetectionApp.__ocr
