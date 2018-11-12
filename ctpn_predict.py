from datetime import datetime

from ctpn import CTPN
import keras.backend as K

from ctpn.lib.utils import get_session

if __name__ == '__main__':
    image_path = "data/demo_02.jpg"                     # 图像位置
    config_path = "config/ctpn-default.json"            # 模型配置路径
    weight_path = "model/weights-ctpnlstm-init.hdf5"    # 模型权重位置

    K.set_session(get_session())

    config = CTPN.load_config(config_path)
    if weight_path is not None:
        config['weight_path'] = weight_path

    ctpn = CTPN(**config)
    start_time = datetime.now()
    ctpn.predict(image_path, output_path="data/demo_encode_02.jpg")
    print(f"cost {(datetime.now() - start_time).microseconds / 1000} ms")
