from datetime import datetime

from densenetocr import DenseNetOCR
from densenetocr.data_loader import load_dict

if __name__ == '__main__':
    dict_file_path = "data/char_std_5990.txt"
    weight_path = "model/weights-densent-init.hdf5"
    config_path = "config/densent-default.json"
    id_to_char = load_dict(dict_file_path, "UTF-8")

    config = DenseNetOCR.load_config(config_path)
    config['weight_path'] = weight_path

    ocr = DenseNetOCR(**config)

    start = datetime.now()
    print(ocr.predict("data/20437812_1996125331.jpg", id_to_char)[0])
    print(f"cost {(datetime.now() - start).microseconds / 1000} ms")
