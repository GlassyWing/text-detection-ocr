import os

import keras.backend as K
import tensorflow as tf
from keras.callbacks import EarlyStopping, TensorBoard

from dlocr.ctpn import CTPN
from dlocr.ctpn import default_ctpn_config_path
from dlocr.ctpn.data_loader import DataLoader
from dlocr.custom import SingleModelCK
from dlocr.custom.callbacks import SGDRScheduler

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-ie", "--initial_epoch", help="初始迭代数", default=0, type=int)
    parser.add_argument("--epochs", help="迭代数", default=20, type=int)
    parser.add_argument("--gpus", help="gpu的数量", default=1, type=int)
    parser.add_argument("--images_dir", help="图像位置", default="E:\data\VOCdevkit\VOC2007\JPEGImages")
    parser.add_argument("--anno_dir", help="标注文件位置", default="E:\data\VOCdevkit\VOC2007\Annotations")
    parser.add_argument("--config_file_path", help="模型配置文件位置",
                        default=default_ctpn_config_path)
    parser.add_argument("--weights_file_path", help="模型初始权重文件位置",
                        default=None)
    parser.add_argument("--save_weights_file_path", help="保存模型训练权重文件位置",
                        default=r'model/weights-ctpnlstm-{epoch:02d}.hdf5')

    args = parser.parse_args()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    K.set_session(session)

    config = CTPN.load_config(args.config_file_path)

    weights_file_path = args.weights_file_path
    if weights_file_path is not None:
        config["weight_path"] = weights_file_path
    config['num_gpu'] = args.gpus

    ctpn = CTPN(**config)

    save_weigths_file_path = args.save_weights_file_path

    if save_weigths_file_path is None:
        try:
            if not os.path.exists("model"):
                os.makedirs("model")
            save_weigths_file_path = "model/weights-ctpnlstm-{epoch:02d}.hdf5"
        except OSError:
            print('Error: Creating directory. ' + "model")

    data_loader = DataLoader(args.anno_dir, args.images_dir)

    checkpoint = SingleModelCK(save_weigths_file_path, model=ctpn.model, save_weights_only=True, monitor='loss')
    earlystop = EarlyStopping(patience=10, monitor='loss')
    log = TensorBoard(log_dir='logs', histogram_freq=0, batch_size=1, write_graph=True, write_grads=False)
    lr_scheduler = SGDRScheduler(min_lr=1e-6, max_lr=1e-4,
                                 initial_epoch=args.initial_epoch,
                                 steps_per_epoch=data_loader.steps_per_epoch,
                                 cycle_length=8,
                                 lr_decay=0.5,
                                 mult_factor=1.2)

    ctpn.train(data_loader.load_data(),
               epochs=args.epochs,
               steps_per_epoch=data_loader.steps_per_epoch,
               callbacks=[checkpoint, earlystop, lr_scheduler],
               initial_epoch=args.initial_epoch)
