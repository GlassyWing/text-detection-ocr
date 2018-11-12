from datetime import datetime

from ctpn import CTPN
import keras.backend as K

from ctpn.lib.utils import get_session

if __name__ == '__main__':
    image_path = "data/demo_02.jpg"
    config_path = "config/ctpn-default.json"
    weight_path = "model/weights-ctpnlstm-init.hdf5"

    K.set_session(get_session())

    config = CTPN.load_config(config_path)
    if weight_path is not None:
        config['weight_path'] = weight_path

    ctpn = CTPN(**config)
    start_time = datetime.now()
    ctpn.predict(image_path, output_path="data/demo_encode_02.jpg")
    print(f"cost {(datetime.now() - start_time).microseconds / 1000} ms")
