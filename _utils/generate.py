# -*- coding: UTF-8 -*-
'''
@Project ：CNN_LSTM
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import os
import torch
import numpy as np
from PIL import Image
from _utils.utils import get_random_data, preprocess_true_boxes

class Generator:
    def __init__(self,
                 annotation_path: str,
                 input_size: tuple,
                 batch_size: int,
                 train_split: float,
                 anchors: np.ndarray,
                 num_classes: int):
        self.annotation_path = annotation_path
        self.input_size = input_size
        self.batch_size = batch_size
        self.train_split = train_split
        self.num_classes = num_classes
        self.anchors = anchors
        self.split_train_val()

    def split_train_val(self):

        with open(self.annotation_path) as f:
            lines = f.readlines()
        np.random.shuffle(lines)

        num_train = int(lines.__len__() * self.train_split)

        self.train_lines = lines[:num_train]
        self.val_lines = lines[num_train:]

    def get_train_len(self):

        if not self.train_lines.__len__() % self.batch_size:
            return self.train_lines.__len__() // self.batch_size
        else:
            return self.train_lines.__len__() // self.batch_size + 1

    def get_val_len(self):

        if not self.val_lines.__len__() % self.batch_size:
            return self.val_lines.__len__() // self.batch_size
        else:
            return self.val_lines.__len__() // self.batch_size + 1

    def generate(self, training=True):

        while True:
            sources, targets = list(), list()
            if training:
                train_lines = self.train_lines
                np.random.shuffle(train_lines)
                for i, line in enumerate(train_lines):
                    image_data, box_data = get_random_data(line, self.input_size)

                    sources.append(image_data)
                    targets.append(box_data)

                    if np.logical_or(np.equal(sources.__len__(), self.batch_size),
                                     np.equal(i, train_lines.__len__()-1)):
                        anno_sources = np.array(sources.copy())
                        anno_targets = preprocess_true_boxes(np.array(targets.copy()), self.input_size,
                                                             self.anchors, self.num_classes)
                        sources.clear()
                        targets.clear()

                        yield anno_sources, anno_targets
            else:
                val_lines = self.val_lines
                np.random.shuffle(val_lines)
                for i, line in enumerate(val_lines):
                    image_data, box_data = get_random_data(line, self.input_size, random=False)

                    sources.append(image_data)
                    targets.append(box_data)

                    if np.logical_or(np.equal(sources.__len__(), self.batch_size),
                                     np.equal(i, val_lines.__len__() - 1)):
                        anno_sources = np.array(sources.copy())
                        anno_targets = preprocess_true_boxes(np.array(targets.copy()), self.input_size,
                                                             self.anchors, self.num_classes)
                        sources.clear()
                        targets.clear()

                        yield anno_sources, anno_targets
