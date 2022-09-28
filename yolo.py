# -*- coding: UTF-8 -*-
'''
@Project ：CNN_LSTM
@File    ：train.py
@IDE     ：PyCharm
@Author  ：XinYi Huang
'''
import torch
import numpy as np
from torch import nn
from PIL import Image, ImageDraw, ImageFont
from net.network import YoloBody
from custom.loss import yolo_loss
from _utils.yolo_utils import yolo_eval
from _utils.utils import calculate_score
from configure import config as cfg


class YOLO:
    def __init__(self,
                 input_shape: tuple,
                 anchors: np.ndarray,
                 classes_names: list,
                 learning_rate: float,
                 score_thresh: float,
                 iou_thresh: float,
                 max_boxes: int,
                 letterbox: bool,
                 weight_decay: float,
                 resume_train: bool,
                 ckpt_path: str):

        self.max_boxes = max_boxes
        self.letterbox = letterbox
        self.iou_thresh = iou_thresh
        self.score_thresh = score_thresh
        self.classes_names = classes_names
        self.num_anchors = anchors.__len__()
        self.num_classes = classes_names.__len__()
        self.learning_rate = learning_rate
        self.anchors = torch.tensor(anchors, dtype=torch.float, device=cfg.device)

        self.model = YoloBody(class_num=classes_names.__len__(),
                              anchor_num=self.num_anchors//2)

        if cfg.device:
            self.model = self.model.to(cfg.device)

        if resume_train:
            try:
                ckpt = torch.load(ckpt_path)
                self.model.load_state_dict(ckpt['state_dict'])
                print("model successfully loaded, loss is {:.3f}".format(ckpt['loss']))
            except FileNotFoundError:
                raise ("please enter the right params path")

        self.weight_decay = weight_decay

        self.optimizer = torch.optim.Adam(lr=self.learning_rate,
                                          params=self.model.parameters())

        self.train_loss, self.val_loss = 0, 0
        self.train_acc, self.val_acc = 0, 0
        self.train_conf_acc, self.val_conf_acc = 0, 0
        self.train_f1_score, self.val_f1_score = 0, 0

    def train(self, sources, targets):

        sources = torch.tensor(sources, dtype=torch.float)
        targets = [torch.tensor(target, dtype=torch.float) for target in targets]
        if cfg.device:
            sources = sources.to(cfg.device)
            targets = [target.to(cfg.device) for target in targets]

        self.optimizer.zero_grad()

        logits = self.model(sources)
        loss = yolo_loss(targets, logits, self.anchors, self.num_classes)
        for weight in self.model.get_weights():
            loss += self.weight_decay * torch.sum(torch.square(weight))

        loss.backward()
        self.optimizer.step()

        self.train_loss += loss.data.item()

        total_nums = [torch.prod(torch.tensor(target[..., 4:5].size())).data.item() for target in targets]
        object_nums = [target[..., 4:5].cpu().sum().data.item() for target in targets]

        prob_confs = [torch.sigmoid(logit[..., 4:5]) for logit in logits]
        prob_confs = [torch.where(torch.greater_equal(prob_conf, torch.tensor(.5)),
                                  torch.ones_like(prob_conf),
                                  torch.zeros_like(prob_conf)) for prob_conf in prob_confs]

        real_confs = [target[..., 4:5] for target in targets]

        correct_conf_nums = [torch.eq(real_conf, prob_conf).float().detach().cpu().sum().data.item()
                             for real_conf, prob_conf in zip(real_confs, prob_confs)]

        self.train_conf_acc += sum(correct_conf_nums)/sum(total_nums)

        object_masks = [real_conf.squeeze(dim=-1).bool() for real_conf in real_confs]

        prob_classes = [logit[..., 5:][mask].argmax(dim=-1) for logit, mask in zip(logits, object_masks)]
        real_classes = [target[..., 5:][mask].argmax(dim=-1) for target, mask in zip(targets, object_masks)]
        correct_class_nums = [torch.eq(real_class, prob_class).float().detach().cpu().sum().data.item()
                              for real_class, prob_class in zip(real_classes, prob_classes)]

        self.train_f1_score += calculate_score(logits, targets, object_masks, self.classes_names.__len__())

        self.train_acc += sum(correct_class_nums)/sum(object_nums)

    def validate(self, sources, targets):

        sources = torch.tensor(sources, dtype=torch.float)
        targets = [torch.tensor(target, dtype=torch.float) for target in targets]
        if cfg.device:
            sources = sources.to(cfg.device)
            targets = [target.to(cfg.device) for target in targets]

        logits = self.model(sources)
        loss = yolo_loss(targets, logits, self.anchors, self.num_classes)
        for weight in self.model.get_weights():
            loss += self.weight_decay * torch.sum(torch.square(weight))

        self.val_loss += loss.data.item()

        total_nums = [torch.prod(torch.tensor(target[..., 4:5].size())).data.item() for target in targets]
        object_nums = [target[..., 4:5].cpu().sum().data.item() for target in targets]

        prob_confs = [torch.sigmoid(logit[..., 4:5]) for logit in logits]
        prob_confs = [torch.where(torch.greater_equal(prob_conf, torch.tensor(.5)),
                                  torch.ones_like(prob_conf),
                                  torch.zeros_like(prob_conf)) for prob_conf in prob_confs]

        real_confs = [target[..., 4:5] for target in targets]

        correct_conf_nums = [torch.eq(real_conf, prob_conf).float().detach().cpu().sum().data.item()
                             for real_conf, prob_conf in zip(real_confs, prob_confs)]

        self.val_conf_acc += sum(correct_conf_nums) / sum(total_nums)

        object_masks = [real_conf.squeeze(dim=-1).bool() for real_conf in real_confs]

        prob_classes = [logit[..., 5:][mask].argmax(dim=-1) for logit, mask in zip(logits, object_masks)]
        real_classes = [target[..., 5:][mask].argmax(dim=-1) for target, mask in zip(targets, object_masks)]
        correct_class_nums = [torch.eq(real_class, prob_class).float().detach().cpu().sum().data.item()
                              for real_class, prob_class in zip(real_classes, prob_classes)]

        self.val_f1_score += calculate_score(logits, targets, object_masks, self.classes_names.__len__())

        self.val_acc += sum(correct_class_nums) / sum(object_nums)

    def generate_sample(self, sources, batch):

        """
        Drawing and labeling
        """
        sources = torch.tensor(sources, dtype=torch.float)
        if cfg.device:
            sources = sources.to(cfg.device)

        logits = self.model(sources)
        image_size = torch.tensor(sources.size()[2:], dtype=torch.float, device=cfg.device)

        out_boxes, out_scores, out_classes = yolo_eval(anchors=self.anchors,
                                                       num_classes=self.num_classes,
                                                       image_shape=image_size,
                                                       yolo_outputs=logits,
                                                       max_boxes=self.max_boxes,
                                                       score_threshold=self.score_thresh,
                                                       iou_threshold=self.iou_thresh,
                                                       letterbox_image=self.letterbox)

        out_boxes = [out_box.detach().cpu().numpy() for out_box in out_boxes]
        out_scores = [out_score.detach().cpu().numpy() for out_score in out_scores]
        out_classes = [out_class.detach().cpu().numpy() for out_class in out_classes]

        index = np.random.choice(np.shape(sources)[0], 1)[0]
        source = sources[index].cpu().numpy().transpose([1, 2, 0])
        image = Image.fromarray(np.uint8(source * 255))

        for i, coordinate in enumerate(out_boxes[index].astype('int')):
            left, top = list(reversed(coordinate[:2]))
            right, bottom = list(reversed(coordinate[2:]))

            font = ImageFont.truetype(font=cfg.font_path,
                                      size=np.floor(4e-2 * image.size[1] + 0.5).astype('int32'))

            label = '{:s}: {:.2f}'.format(self.classes_names[out_classes[index][i]],
                                          out_scores[index][i])

            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            draw.rectangle(list(reversed(coordinate[:2])) + list(reversed(coordinate[2:])),
                           outline=cfg.rect_color, width=int(2 * cfg.thickness))

            draw.text(text_origin, str(label, 'UTF-8'),
                      fill=cfg.font_color, font=font)
            del draw
        image.save(cfg.sample_path.format(batch), quality=95, subsampling=0)
