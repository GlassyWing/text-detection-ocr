import os

from dlocr.densenet.core import DenseNetOCR

default_densenet_config_path = os.path.join(os.getcwd(), os.path.dirname(__file__), "../config/densent-default.json")
default_dict_path = os.path.join(os.getcwd(), os.path.dirname(__file__), "../dictionary/dict.json")

__densenet = None


def get_or_create(densenet_weight_path, densenet_config_path=default_densenet_config_path):
    global __densenet
    if __densenet is None:
        config = DenseNetOCR.load_config(densenet_config_path)
        __densenet = DenseNetOCR(**config, weight_path=densenet_weight_path)
    return __densenet
