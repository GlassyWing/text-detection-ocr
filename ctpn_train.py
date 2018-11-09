from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, Callback

from ctpn import CTPN
from lib.utils import gen_sample
import numpy as np
import keras.backend as K


class HistoryCache:

    def __init__(self, his_len=10):
        self.history = [0] * his_len
        self.history_len = his_len
        self.cursor = 0

    def put(self, value):
        self.history[self.cursor] = value
        self.cursor += 1
        if self.cursor >= self.history_len:
            self.cursor = 0

    def mean(self):
        return np.array(self.history).mean()


class LRScheduler(Callback):

    def __init__(self, schedule, watch, watch_his_len=10):
        super().__init__()
        self.schedule = schedule
        self.watch = watch
        self.history_cache = HistoryCache(watch_his_len)

    def on_epoch_begin(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)

    def on_epoch_end(self, epoch, logs=None):
        lr = float(K.get_value(self.model.optimizer.lr))
        watch_value = logs.get(self.watch)
        if watch_value is None:
            raise ValueError(f"Watched value '{self.watch}' don't exist")

        if watch_value > self.history_cache.mean():
            lr = self.schedule(epoch, lr)
            K.set_value(self.model.optimizer.lr, lr)

        self.history_cache.put(watch_value)


if __name__ == '__main__':
    gen = gen_sample("E:\data\VOCdevkit\VOC2007\Annotations", "E:\data\VOCdevkit\VOC2007\JPEGImages")

    checkpoint = ModelCheckpoint(r'model\weights-ctpnlstm-{epoch:02d}.hdf5', save_weights_only=True)
    earlystop = EarlyStopping(patience=10)
    lr_scheduler = LRScheduler(lambda epoch, lr: lr / 2, watch="loss", watch_his_len=1)

    ctpn = CTPN(vgg_trainable=True)
    ctpn.train(gen, epochs=20, steps_per_epoch=2, callbacks=[checkpoint, earlystop, lr_scheduler])
