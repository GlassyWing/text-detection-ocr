from datetime import datetime

from ctpn import CTPN
import keras.backend as K

from ctpn.lib.utils import get_session

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", help="图像位置")
    parser.add_argument("--config_file_path", help="模型配置文件位置",
                        default="config/ctpn-default.json")
    parser.add_argument("--weights_file_path", help="模型权重文件位置",
                        default="model/weights-ctpnlstm-init.hdf5")
    parser.add_argument("--output_file_path", help="标记文件保存位置",
                        default=None)

    args = parser.parse_args()

    K.set_session(get_session())

    image_path = args.image_path  # 图像位置
    config_path = args.config_file_path  # 模型配置路径
    weight_path = args.weights_file_path  # 模型权重位置
    output_file_path = args.output_file_path  # 保存标记文件位置

    config = CTPN.load_config(config_path)
    if weight_path is not None:
        config['weight_path'] = weight_path

    ctpn = CTPN(**config)
    start_time = datetime.now()
    ctpn.predict(image_path, output_path=output_file_path)
    print(f"cost {(datetime.now() - start_time).microseconds / 1000} ms")
