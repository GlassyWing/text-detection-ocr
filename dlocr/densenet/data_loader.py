from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from pathlib import Path


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


def load_dict(dict_file_path, encoding="utf-8", blank_first=True):
    with open(dict_file_path, encoding=encoding, mode='r') as f:
        chars = list(map(lambda char: char.strip('\r\n'), f.readlines()))

    if blank_first:
        chars = chars[1:] + ['blank']

    dict = {i: v for i, v in enumerate(chars)}

    return dict


class DataLoader:

    def __init__(self, dict_file_path,
                 labeled_file_path,
                 images_dir,
                 encoding='utf-8',
                 blank_first=True,
                 batch_size=64,
                 maxlen=10,
                 image_shape=(32, 280)):
        self.images_dir = images_dir
        self.blank_first = blank_first
        self.maxlen = maxlen
        self.image_shape = image_shape
        self.id_to_dict = self.__load_dict(dict_file_path, encoding)
        self.num_classes = len(self.id_to_dict)
        self.batch_size = batch_size
        self.image_label = self.__load_labeled_file(labeled_file_path, encoding)
        self.image_files = list(self.image_label.keys())
        self.total_size = len(self.image_files)
        self.random_uniform_selector = _RandomUniformSelector(self.total_size)
        self.steps_per_epoch = self.total_size // self.batch_size

    def __load_dict(self, dict_file_path, encoding='utf-8'):
        with open(dict_file_path, encoding=encoding, mode='r') as f:
            chars = list(map(lambda char: char.strip('\r\n'), f.readlines()))

        if self.blank_first:
            chars = chars[1:] + ['blank']

        return {i: v for i, v in enumerate(chars)}

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
            label = np.ones([self.maxlen]) * self.num_classes

            if self.blank_first:
                label[0: len(image_label)] = [int(i) - 1 for i in image_label]
            else:
                label[0: len(image_label)] = [int(i) for i in image_label]

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
    train_data_loader = DataLoader(images_dir="E:/data/images",
                                   dict_file_path="../data/char_std_5990.txt",
                                   labeled_file_path="../data/test.txt",
                                   image_shape=(32, 280),
                                   encoding="UTF-8",
                                   blank_first=True)
    # x, label = next(train_data_loader.load_data())
    print(train_data_loader.total_size)
