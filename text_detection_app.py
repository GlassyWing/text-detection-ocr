from datetime import datetime
from math import *

import cv2
import numpy as np
from PIL import Image

from ctpn import CTPN
from densenetocr import DenseNetOCR
from densenetocr.data_loader import load_dict


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
    return text


def model(ctpn, ocr, id_to_char, image_path, adjust):
    text_recs, img = ctpn.predict(image_path, mode=2)
    text_recs = sort_box(text_recs)
    results = []

    for index, rec in enumerate(text_recs):
        text = single_text_detect(rec, ocr, id_to_char, img, adjust)

        if text is not None and len(text) > 0:
            results.append((rec, text))  # 识别文字

    return results


class TextDetectionApp:

    def __init__(self,
                 ctpn_weight_path,
                 densenet_weight_path,
                 dict_path,
                 ctpn_config_path=None,
                 densenet_config_path=None):

        self.id_to_char = load_dict(dict_path, encoding="utf-8")

        if ctpn_config_path is not None:
            ctpn_config = CTPN.load_config(ctpn_config_path)
            ctpn_config["weight_path"] = ctpn_weight_path
            self.ctpn = CTPN(**ctpn_config)
        else:
            self.ctpn = CTPN()
        if densenet_config_path is not None:
            densenet_config = DenseNetOCR.load_config(densenet_config_path)
            densenet_config["weight_path"] = densenet_weight_path
            self.ocr = DenseNetOCR(**densenet_config)
        else:
            self.ocr = DenseNetOCR(num_classes=len(self.id_to_char))

    def detect(self, image_path, adjust):
        return model(self.ctpn, self.ocr, self.id_to_char, image_path, adjust)


if __name__ == '__main__':
    app = TextDetectionApp(ctpn_weight_path="model/weights-ctpnlstm-init.hdf5",
                           densenet_weight_path="model/weights-densent-init.hdf5",
                           dict_path="data/char_std_5990.txt",
                           ctpn_config_path="config/ctpn-default.json",
                           densenet_config_path="config/densent-default.json")
    start_time = datetime.now()
    for line_no, line in app.detect("data/demo.jpg", False):
        print(line)
    print(f"cost {(datetime.now() - start_time).microseconds / 1000}ms")
