import os
import torch
from _utils.utils import get_classes, get_anchors


# ===annotation===
xml_root = 'coco数据集标注文件根目录'
image_root = 'coco数据集根目录'


# ===generator===
annotation_path = '.\\annotations\\get_annotation.py获取的txt文件路径(置于model_data中)'
classes_path = '定义检测目标的类别的txt文件路径(置于.\\model_data中)'
anchors_path = '.\\annotations\\kmeans_for_anchors.py获取的先验框尺寸的txt文件路径(置于.\\model_data中)'
train_split = 0.7
input_size = (416, 416)
letterbox = False

# ===model===
anchors = get_anchors(anchors_path)
class_names = get_classes(classes_path)

# ===training===
Epoches = 150
batch_size = 16
learning_rate = 3e-4
weight_decay = 5e-4
warmup_learning_rate = 1e-5
min_learning_rate = 1e-7
remain_train = True
cosine_scheduler = False
device = torch.device('cuda') if torch.cuda.is_available() else None
per_sample_interval = 100
ckpt_path = '.\\checkpoint'

# ===prediction===
iou = 0.5
score = 0.6
max_boxes = 100
font_color = (0, 255, 0)
rect_color = (0, 0, 255)
thickness = 0.5
font_path = '.\\font\\simhei.ttf'
sample_path = ".\\sample\\Batch{}.jpg"

defaults = {
    "ckpt_path": '模型文件根目录',
    "anchors_path": 'anchors文件根目录',
    "classes_path": 'classes文件根目录',
    "score": 0.6,
    "iou": 0.5,
    "max_boxes": 100,
    "image_size": (416, 416), # (height, width)
    "letterbox_image": False,
    "device": torch.device('cuda') if torch.cuda.is_available() else None
}
