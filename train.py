# -*- coding: UTF-8 -*-
'''
@Project ：CNN_LSTM
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import os
import torch
from torch.nn import functional as F
from yolo import YOLO
from _utils.generate import Generator
from configure import config as cfg

if __name__ == '__main__':

    Yolo = YOLO(input_shape=cfg.input_size,
                anchors=cfg.anchors,
                classes_names=cfg.class_names,
                learning_rate=cfg.learning_rate,
                score_thresh=cfg.score,
                iou_thresh=cfg.iou,
                max_boxes=cfg.max_boxes,
                letterbox=cfg.letterbox,
                weight_decay=cfg.weight_decay,
                resume_train=cfg.remain_train,
                ckpt_path=cfg.ckpt_path + "\\模型文件")

    data_gen = Generator(annotation_path=cfg.annotation_path,
                         input_size=cfg.input_size,
                         batch_size=cfg.batch_size,
                         train_split=cfg.train_split,
                         anchors=cfg.anchors,
                         num_classes=cfg.class_names.__len__())

    train_gen = data_gen.generate(training=True)
    validate_gen = data_gen.generate(training=False)

    for epoch in range(cfg.Epoches):
        for i in range(data_gen.get_train_len()):

            sources, targets = next(train_gen)
            Yolo.train(sources, targets)
            if not (i + 1) % cfg.per_sample_interval:
                Yolo.generate_sample(sources, i+1)

        print('Epoch{:0>3d} '
              'train loss is {:.3f} '
              'train acc is {:.3f} '
              'train conf acc is {:.3f} '
              'train f1 score is {:.3f}'.format(epoch+1,
                                                Yolo.train_loss / (i + 1),
                                                Yolo.train_acc / (i + 1) * 100,
                                                Yolo.train_conf_acc / (i + 1) * 100,
                                                Yolo.train_f1_score / (i + 1) * 100))

        torch.save({'state_dict': Yolo.model.state_dict(),
                    'loss': Yolo.train_loss / (i + 1),
                    'acc': Yolo.train_acc / (i + 1) * 100},
                   cfg.ckpt_path + '\\Epoch{:0>3d}_train_loss{:.3f}_train_acc{:.3f}.pth.tar'.format(
                       epoch + 1, Yolo.train_loss / (i + 1), Yolo.train_acc / (i + 1) * 100))
        
        Yolo.train_loss = 0
        Yolo.train_acc = 0
        Yolo.train_conf_acc = 0
        Yolo.train_f1_score = 0

        for i in range(data_gen.get_val_len()):
            sources, targets = next(validate_gen)
            Yolo.validate(sources, targets)

        print('Epoch{:0>3d} '
              'validate loss is {:.3f} '
              'validate acc is {:.3f} '
              'validate conf acc is {:.3f} '
              'validate f1 score is {:.3f}'.format(epoch+1,
                                                   Yolo.val_loss / (i + 1),
                                                   Yolo.val_acc / (i + 1) * 100,
                                                   Yolo.val_conf_acc / (i + 1) * 100,
                                                   Yolo.val_f1_score / (i + 1) * 100))

        Yolo.val_loss = 0
        Yolo.val_acc = 0
        Yolo.val_conf_acc = 0
        Yolo.val_f1_score = 0
