import os

from dlocr.densenet.core import DenseNetOCR
from dlocr.densenet.data_loader import load_dict

default_densenet_weight_path = os.path.join(os.getcwd(), os.path.dirname(__file__), "../weights/weights-densent-init.hdf5")
default_densenet_config_path = os.path.join(os.getcwd(), os.path.dirname(__file__), "../config/densent-default.json")
default_dict_path = os.path.join(os.getcwd(), os.path.dirname(__file__), "../dictionary/char_std_5990.txt")

_config = DenseNetOCR.load_config(default_densenet_config_path)
densenet = DenseNetOCR(**_config, weight_path=default_densenet_weight_path)


