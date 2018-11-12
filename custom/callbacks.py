import keras.backend as K
import numpy as np
from keras.callbacks import Callback, ModelCheckpoint


class HistoryCache:

    def __init__(self, his_len=10):
        self.history = [0] * his_len
        self.history_len = his_len
        self.cursor = 0
        self.len = 0

    def put(self, value):
        self.history[self.cursor] = value
        self.cursor += 1
        if self.cursor >= self.history_len:
            self.cursor = 0
        if self.len + 1 <= self.history_len:
            self.len += 1

    def mean(self):
        return np.array(self.history[0: self.len]).mean()


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

        self.history_cache.put(watch_value)

        if watch_value > self.history_cache.mean():
            lr = self.schedule(epoch, lr)
            print(f"Update learning rate: {lr}")
            K.set_value(self.model.optimizer.lr, lr)


class SingleModelCK(ModelCheckpoint):
    """
    用于解决在多gpu下训练保存的权重无法应用于单gpu的情况
    """

    def __init__(self, filepath, model, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super().__init__(filepath=filepath, monitor=monitor, verbose=verbose,
                         save_weights_only=save_weights_only,
                         save_best_only=save_best_only,
                         mode=mode, period=period)
        self.model = model

    def set_model(self, model):
        pass
