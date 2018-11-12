import argparse
from datetime import datetime

from ctpn.lib.utils import get_session
from densenetocr import DenseNetOCR
from densenetocr.data_loader import load_dict
import keras.backend as K

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", help="图像位置")
    parser.add_argument("--dict_file_path", help="字典文件位置", default="data/char_std_5990.txt")
    parser.add_argument("--config_file_path", help="模型配置文件位置",
                        default="config/densent-default.json")
    parser.add_argument("--weights_file_path", help="模型权重文件位置",
                        default="model/weights-densent-init.hdf5")

    args = parser.parse_args()

    K.set_session(get_session(0.4))

    image_path = args.image_path
    dict_file_path = args.dict_file_path  # 字典文件位置
    weight_path = args.weights_file_path  # 权重文件位置
    config_path = args.config_file_path  # 模型配置文件位置

    id_to_char = load_dict(dict_file_path, "UTF-8")

    config = DenseNetOCR.load_config(config_path)
    config['weight_path'] = weight_path

    ocr = DenseNetOCR(**config)

    start = datetime.now()
    print(ocr.predict(image_path, id_to_char)[0])
    print(f"cost {(datetime.now() - start).microseconds / 1000} ms")
