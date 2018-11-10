from keras.callbacks import ModelCheckpoint, EarlyStopping

from ctpn import CTPN
from ctpn.lib.utils import gen_sample
from custom import LRScheduler

if __name__ == '__main__':
    gen = gen_sample("E:\data\VOCdevkit\VOC2007\Annotations", "E:\data\VOCdevkit\VOC2007\JPEGImages")

    checkpoint = ModelCheckpoint(r'model\weights-ctpnlstm-{epoch:02d}.hdf5', save_weights_only=True)
    earlystop = EarlyStopping(patience=10)
    lr_scheduler = LRScheduler(lambda epoch, lr: lr / 2, watch="loss", watch_his_len=1)

    ctpn = CTPN(vgg_trainable=True)
    ctpn.train(gen, epochs=20, steps_per_epoch=2, callbacks=[checkpoint, earlystop, lr_scheduler])
