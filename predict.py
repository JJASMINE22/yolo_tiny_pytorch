# -*- coding: UTF-8 -*-
'''
@Project ：CNN_LSTM
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import torch
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from net.network import YoloBody
from _utils.yolo_utils import yolo_eval
from _utils.utils import get_classes, get_anchors, letterbox_image
from configure import config as cfg

class YoloEval:
    def __init__(self,
                 **kwargs):
        self.__dict__.update(kwargs)
        self.class_names = get_classes(self.classes_path)
        self.anchors = get_anchors(self.anchors_path)
        self.anchors = torch.from_numpy(self.anchors).to(self.device)

        self.model = YoloBody(class_num=self.class_names.__len__(),
                              anchor_num=self.anchors.__len__()//2)
        self.model = self.model.to(self.device)

        if hasattr(self, 'ckpt_path'):
            try:
                ckpt = torch.load(self.ckpt_path)
                self.model.load_state_dict(ckpt['state_dict'])
                # self.model.eval()
                print("model successfully loaded, loss is {:.3f} acc is {:.3f}%".format(ckpt['loss'], ckpt['acc']))
            except FileNotFoundError:
                raise ("Invalid params, check model file again !")
        else:
            raise ("Add model file !")

    def detect_image(self, image):
        if self.letterbox_image:
            boxed_image = letterbox_image(image, tuple(reversed(self.image_size)))
        else:
            boxed_image = image.convert('RGB')
            boxed_image = boxed_image.resize(reversed(self.image_size), Image.BICUBIC)

        image_size = torch.from_numpy(np.array(image.size)).flip(dims=[0]).to(self.device)
        image_data = np.array(boxed_image, dtype='float32')
        image_data = image_data.transpose((2, 0, 1))/255.
        image_data = image_data[np.newaxis, ...]
        image_data = torch.from_numpy(image_data).to(self.device)

        logits = self.model(image_data)
        boxes, scores, classes = yolo_eval(anchors=self.anchors,
                                           num_classes=self.class_names.__len__(),
                                           image_shape=image_size,
                                           yolo_outputs=logits,
                                           max_boxes=self.max_boxes,
                                           score_threshold=self.score,
                                           iou_threshold=self.iou,
                                           letterbox_image=self.letterbox_image)

        return boxes, scores, classes

    def assign(self, image):

        boxes, scores, classes = self.detect_image(image)
        boxes, scores, classes = boxes[0], scores[0], classes[0]

        for i, (coordinate, confidence, class_prob) in enumerate(zip(boxes, scores, classes)):

            coordinate = coordinate.detach().cpu().numpy().astype('int')
            confidence = confidence.detach().cpu().numpy()
            class_prob = class_prob.detach().cpu().numpy()

            top, left= coordinate[:2].tolist()
            bottom, right = coordinate[2:].tolist()

            font = ImageFont.truetype(font=cfg.font_path,
                                      size=np.floor(4e-2 * image.size[1] + 0.5).astype('int32'))

            label = '{:s}: {:.2f}'.format(self.class_names[class_prob], confidence)

            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            draw.rectangle([left, top, right, bottom],
                           outline=cfg.rect_color, width=int(2 * cfg.thickness))
            draw.text(text_origin, str(label, 'UTF-8'),
                      fill=cfg.font_color, font=font)

            del draw
        image.show()

