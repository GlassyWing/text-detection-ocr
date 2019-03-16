from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from pathlib import Path

from keras_preprocessing.text import Tokenizer
from dlocr.utils import load_dictionary


class _RandomUniformSelector:

    def __init__(self, total, start=0):
        self.total = total
        self.idxes = [i for i in range(total)]
        np.random.shuffle(self.idxes)
        self.cursor = start

    def next(self, batch_size):
        if self.cursor + batch_size > self.total:
            r_n = []
            r_n_1 = self.idxes[self.cursor:self.total]
            self.cursor = self.cursor + batch_size - self.total
            r_n_2 = self.idxes[0:self.cursor]
            r_n.extend(r_n_1)
            r_n.extend(r_n_2)
        else:
            r_n = self.idxes[self.cursor:self.cursor + batch_size]
            self.cursor = self.cursor + batch_size
        return r_n


class DataLoader:

    def __init__(self, dict_file_path,
                 labeled_file_path,
                 images_dir,
                 encoding='utf-8',
                 batch_size=64,
                 maxlen=11,
                 image_shape=(32, 280)):
        self.images_dir = images_dir
        self.maxlen = maxlen
        self.image_shape = image_shape
        self.tokenizer: Tokenizer = load_dictionary(dict_file_path, encoding=encoding)
        self.num_classes = len(self.tokenizer.word_index)
        self.batch_size = batch_size
        self.image_label = self.__load_labeled_file(labeled_file_path, encoding)
        self.image_files = list(self.image_label.keys())
        self.total_size = len(self.image_files)
        self.random_uniform_selector = _RandomUniformSelector(self.total_size)
        self.steps_per_epoch = self.total_size // self.batch_size

    def __load_labeled_file(self, labeled_file_path, encoding='utf-8'):
        dic = {}
        with open(labeled_file_path, encoding=encoding, mode="r") as f:
            for line in f.readlines():
                line = line.strip('\n')
                line = line.split(' ')
                dic[line[0]] = line[1:]
        return dic

    def load_data(self):
        image_files = np.array(self.image_files)

        def load_single_example(image_file_path, image_label):
            # rescale to [-1, 1]
            img = np.array(Image.open(Path(self.images_dir) / image_file_path).convert('L')) / 255.0 - 0.5
            img = np.expand_dims(img, axis=2)
            label_len = np.array([len(image_label)])
            input_len = np.array([self.image_shape[1] // 8])

            label = np.ones([self.maxlen], dtype=np.int32) * (self.num_classes - 1)

            label[: len(image_label)] = [int(i) for i in image_label]

            return img, label_len, input_len, label

        while True:
            shuffled_image_files = image_files[self.random_uniform_selector.next(self.batch_size)]
            image_labels = [(image_file, self.image_label[image_file]) for image_file in shuffled_image_files]

            imgs = []
            label_lens = []
            input_lens = []
            labels = []
            with ThreadPoolExecutor() as executor:
                for img, label_len, input_len, label in executor.map(lambda t: load_single_example(*t), image_labels):
                    imgs.append(img)
                    label_lens.append(label_len)
                    input_lens.append(input_len)
                    labels.append(label)

            inputs = {'the_input': np.array(imgs),
                      'the_labels': np.array(labels),
                      'input_length': np.array(input_lens),
                      'label_length': np.array(label_lens),
                      }
            outputs = {'ctc': np.zeros([self.batch_size])}

            yield inputs, outputs


if __name__ == '__main__':
    train_data_loader = DataLoader(images_dir="G:/data/text-recognition/chinese_imgs/default",
                                   dict_file_path="../dictionary/dict.json",
                                   labeled_file_path="G:/data/text-recognition/chinese_imgs/default/tmp_labels_train.txt",
                                   image_shape=(32, 280),
                                   encoding="UTF-8")
    x, label = next(train_data_loader.load_data())
    print(x['the_input'].shape)
    print(x['the_labels'][4])
    print(train_data_loader.tokenizer.sequences_to_texts([x['the_labels'][4]]))
    print(train_data_loader.total_size)
