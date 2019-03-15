# -*- coding: utf-8 -*-
import argparse
import time

import keras.backend as K

from dlocr import default_dict_path, default_densenet_config_path, default_densenet_weight_path, \
    default_ctpn_config_path, \
    default_ctpn_weight_path, get_session, TextDetectionApp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", help="图像位置", required=True)
    parser.add_argument("--dict_file_path", help="字典文件位置", default=default_dict_path)
    parser.add_argument("--densenet_config_path", help="densenet模型配置文件位置",
                        default=default_densenet_config_path)
    parser.add_argument("--ctpn_config_path", help="ctpn模型配置文件位置",
                        default=default_ctpn_config_path)
    parser.add_argument("--ctpn_weight_path", help="ctpn模型权重文件位置",
                        default=default_ctpn_weight_path)
    parser.add_argument("--densenet_weight_path", help="densenet模型权重文件位置",
                        default=default_densenet_weight_path)
    parser.add_argument("--adjust", help="是否对图像进行适当裁剪",
                        default=True, type=bool)

    args = parser.parse_args()

    K.set_session(get_session())

    app = TextDetectionApp(ctpn_weight_path=args.ctpn_weight_path,
                           densenet_weight_path=args.densenet_weight_path,
                           dict_path=args.dict_file_path,
                           ctpn_config_path=args.ctpn_config_path,
                           densenet_config_path=args.densenet_config_path)
    start_time = time.time()
    _, texts = app.detect(args.image_path, args.adjust)
    print('\n'.join(texts))
    print("cost", (time.time() - start_time) * 1000, "ms")
