from keras.callbacks import EarlyStopping, TensorBoard

from ctpn import CTPN
from ctpn.data_loader import DataLoader
from ctpn.lib.utils import get_session
from custom import LRScheduler, SingleModelCK
import keras.backend as K

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-ie", "--initial_epoch", help="初始迭代数", default=0, type=int)
    parser.add_argument("--epochs", help="迭代数", default=20, type=int)
    parser.add_argument("--gpus", help="gpu的数量", default=1, type=int)
    parser.add_argument("--images_dir", help="图像位置", default="E:\data\VOCdevkit\VOC2007\JPEGImages")
    parser.add_argument("--anno_dir", help="标注文件位置", default="E:\data\VOCdevkit\VOC2007\Annotations")
    parser.add_argument("--config_file_path", help="模型配置文件位置",
                        default="config/ctpn-default.json")
    parser.add_argument("--weights_file_path", help="模型初始权重文件位置",
                        default=None)

    args = parser.parse_args()

    K.set_session(get_session(0.8))
    config = CTPN.load_config(args.config_file_path)

    weights_file_path = args.weights_file_path
    if weights_file_path is not None:
        config["weight_path"] = weights_file_path
    config['num_gpu'] = args.gpus

    ctpn = CTPN(**config)

    data_loader = DataLoader(args.anno_dir, args.images_dir)

    checkpoint = SingleModelCK(r'model/weights-ctpnlstm-{epoch:02d}.hdf5', model=ctpn.model, save_weights_only=True)
    earlystop = EarlyStopping(patience=10)
    log = TensorBoard(log_dir='logs', histogram_freq=0, batch_size=1, write_graph=True, write_grads=False)
    lr_scheduler = LRScheduler(lambda epoch, lr: lr / 2, watch="loss", watch_his_len=20)

    ctpn.train(data_loader.load_data(),
               epochs=args.epochs,
               steps_per_epoch=data_loader.steps_per_epoch,
               callbacks=[checkpoint, earlystop, lr_scheduler],
               initial_epoch=args.initial_epoch)
