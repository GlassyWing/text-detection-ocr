import argparse
import time

import keras.backend as K

from dlocr.ctpn.lib.utils import get_session
from dlocr.densenet import default_densenet_weight_path, default_densenet_config_path, default_dict_path
from dlocr.densenet import get_or_create
from dlocr.densenet.data_loader import load_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", help="图像位置")
    parser.add_argument("--dict_file_path", help="字典文件位置", default=default_dict_path)
    parser.add_argument("--config_file_path", help="模型配置文件位置",
                        default=default_densenet_config_path)
    parser.add_argument("--weights_file_path", help="模型权重文件位置",
                        default=default_densenet_weight_path)

    args = parser.parse_args()

    K.set_session(get_session(0.4))

    image_path = args.image_path
    dict_file_path = args.dict_file_path  # 字典文件位置
    weight_path = args.weights_file_path  # 权重文件位置
    config_path = args.config_file_path  # 模型配置文件位置

    id_to_char = load_dict(dict_file_path, "UTF-8")

    if weight_path is not None:
        densenet = get_or_create(config_path, weight_path)
    else:
        densenet = get_or_create(config_path)

    start = time.time()
    print(densenet.predict(image_path, id_to_char)[0])
    print("cost ",(time.time() - start) * 1000," ms")
