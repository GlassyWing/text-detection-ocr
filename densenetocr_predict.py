from datetime import datetime

from densenetocr import DenseNetOCR
from densenetocr.data_loader import load_dict

if __name__ == '__main__':
    dict_file_path = "data/char_std_5990.txt"       # 字典文件位置
    weight_path = "model/weights-densent-init.hdf5" # 权重文件位置
    config_path = "config/densent-default.json"     # 模型配置文件位置
    id_to_char = load_dict(dict_file_path, "UTF-8")

    config = DenseNetOCR.load_config(config_path)
    config['weight_path'] = weight_path

    ocr = DenseNetOCR(**config)

    start = datetime.now()
    print(ocr.predict("data/20437812_1996125331.jpg", id_to_char)[0])
    print(f"cost {(datetime.now() - start).microseconds / 1000} ms")
