#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：labelme-voc 
@File    ：get_annotation.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
@Date    ：2021/5/10 10:54 
'''

import os
import xml.etree.ElementTree as ET
from annotations import cfg


classes = [class_name.strip() for class_name in open(cfg.classes_path, 'r').readlines()]


def convert_annotation(xml_path, image_path, anno_text):
    in_file = open(xml_path, encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()

    counter = 0
    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') != None:
            difficult = obj.find('difficult').text

        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        if not counter:
            anno_text.write(image_path)
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (round(float(xmlbox.find('xmin').text)), round(float(xmlbox.find('ymin').text)),
             round(float(xmlbox.find('xmax').text)), round(float(xmlbox.find('ymax').text)))
        anno_text.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
        counter += 1
    if counter:
        anno_text.write('\n')


if __name__ == '__main__':

    anno_text = open(cfg.annotation_path, 'w')
    for root, dirs, files in os.walk(cfg.xml_root):
        for file in files:
            xml_path = os.path.join(cfg.xml_root, file)
            image_path = os.path.join(cfg.image_root, file.split('.')[0] + '.jpg')
            convert_annotation(xml_path, image_path, anno_text)
    anno_text.close()
