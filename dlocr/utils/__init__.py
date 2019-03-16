import json
from keras_preprocessing.text import tokenizer_from_json


def load_dictionary(dict_path, encoding="utf-8"):
    with open(dict_path, mode="r", encoding=encoding) as file:
        return tokenizer_from_json(json.load(file))