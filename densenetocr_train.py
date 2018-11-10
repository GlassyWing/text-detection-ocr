import keras.backend as K
from keras.callbacks import EarlyStopping, TensorBoard

from custom import LRScheduler, SingleModelCK
from densenetocr import DenseNetOCR
from densenetocr.data_loader import DataLoader
from ctpn.lib import utils

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-ie", "--initial_epoch", help="初始迭代数", default=0, type=int)
    parser.add_argument("-bs", "--batch_size", help="小批量处理大小", default=64, type=int)
    parser.add_argument("--epochs", help="迭代数", default=20, type=int)
    parser.add_argument("--gpus", help="gpu的数量", default=1, type=int)
    parser.add_argument("--images_dir", help="训练图像位置", default="/home/sunsheenai/application/data/OCR/images")
    parser.add_argument("--dict_file_path", help="字典文件位置",
                        default="/home/sunsheenai/application/data/OCR/char_std_5990.txt")
    parser.add_argument("--train_file_path", help="训练文件位置",
                        default="/home/sunsheenai/application/data/OCR/train.txt")
    parser.add_argument("--test_file_path", help="测试文件位置",
                        default="/home/sunsheenai/application/data/OCR/test.txt")
    parser.add_argument("--config_file_path", help="模型配置文件位置",
                        default="config/densent-default.json")
    parser.add_argument("--weights_file_path", help="模型初始权重文件位置",
                        default=None)

    args = parser.parse_args()

    K.set_session(utils.get_session(0.8))

    batch_size = args.batch_size

    encoding = "UTF-8"
    initial_epoch = args.initial_epoch

    # 载入模型配置文件
    config = DenseNetOCR.load_config(args.config_file_path)
    weights_file_path = args.weights_file_path
    gpus = args.gpus
    config['num_gpu'] = gpus

    # 载入初始权重
    if weights_file_path is not None:
        config["weight_path"] = weights_file_path

    # 载入训练数据
    images_dir = args.images_dir
    dict_file_path = args.dict_file_path
    train_labeled_file_path = args.train_file_path
    test_labeled_file_path = args.test_file_path

    # 测试
    # images_dir = "E:\data\images"
    # dict_file_path = "data/char_std_5990.txt"
    # train_labeled_file_path = "data/train.txt"
    # test_labeled_file_path = "data/test.txt"

    train_data_loader = DataLoader(images_dir=images_dir,
                                   dict_file_path=dict_file_path,
                                   labeled_file_path=train_labeled_file_path,
                                   image_shape=(32, 280),
                                   encoding=encoding,
                                   maxlen=config['maxlen'],
                                   batch_size=batch_size,
                                   blank_first=True)

    valid_data_loader = DataLoader(images_dir=images_dir,
                                   dict_file_path=dict_file_path,
                                   labeled_file_path=test_labeled_file_path,
                                   image_shape=(32, 280),
                                   encoding=encoding,
                                   batch_size=batch_size,
                                   maxlen=config['maxlen'],
                                   blank_first=True)

    ocr = DenseNetOCR(**config)

    checkpoint = SingleModelCK(r'model/weights-densent-{epoch:02d}.hdf5',
                               model=ocr.model,
                               save_weights_only=True)

    earlystop = EarlyStopping(patience=10)
    log = TensorBoard(log_dir='logs', histogram_freq=0, batch_size=train_data_loader.batch_size,
                      write_graph=True,
                      write_grads=False)

    # 观测ctc损失的值，一旦损失回升，将学习率缩小一半
    lr_scheduler = LRScheduler(lambda _, lr: lr / 2, watch="loss", watch_his_len=1)

    ocr.train(epochs=args.epochs,
              train_data_loader=train_data_loader,
              valid_data_loader=valid_data_loader,
              callbacks=[earlystop, checkpoint, log, lr_scheduler],
              initial_epoch=initial_epoch)
