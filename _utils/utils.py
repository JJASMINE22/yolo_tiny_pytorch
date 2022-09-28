import os
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def rand(a=0.0, b=1.0):
    return np.random.rand()*(b-a) + a

def get_random_data(annotation_line, input_shape, max_boxes=100, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
    '''random preprocessing for real-time data augmentation'''
    line = annotation_line.split()
    image = Image.open(os.path.join('C:\\DATASET\\COCO\\Images', line[0].split('\\')[-1]))  # line[0]
    iw, ih = image.size
    h, w = input_shape
    box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

    if not random:
        # resize image
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        dx = (w-nw)//2
        dy = (h-nh)//2

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image, np.float32) / 255.
        image_data = np.clip(image_data, 0., 1.)
        image_data = image_data.transpose([2, 0, 1])

        # correct boxes
        box_data = np.zeros((max_boxes, 5))
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]]*nw/iw + dx
            box[:, [1, 3]] = box[:, [1, 3]]*nh/ih + dy
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
            if len(box) > max_boxes: box = box[:max_boxes]
            box_data[:len(box)] = box

        return image_data, box_data
        
    # 对图像进行缩放并且进行长和宽的扭曲
    new_ar = w/h * rand(1-jitter, 1+jitter)/rand(1-jitter, 1+jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale*h)
        nw = int(nh*new_ar)
    else:
        nw = int(scale*w)
        nh = int(nw/new_ar)
    image = image.resize((nw, nh), Image.BICUBIC)

    # 将图像多余的部分加上灰条
    dx = int(rand(0, w-nw))
    dy = int(rand(0, h-nh))
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # 翻转图像
    flip = rand() < .5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # 色域扭曲
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1/rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1/rand(1, val)
    image = np.array(image, np.float32) / 255.
    image = np.clip(image, 0., 1.)
    x = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    x[..., 0] += hue*360
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x[:, :, 0] > 360, 0] = 360
    x[:, :, 1:][x[:, :, 1:] > 1] = 1
    x[x < 0] = 0
    image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)  # numpy array, 0 to 1
    image_data = image_data.transpose([2, 0, 1])

    # 将box进行调整
    box_data = np.zeros((max_boxes, 5))
    if len(box) > 0:
        np.random.shuffle(box)
        box[:, [0, 2]] = box[:, [0, 2]]*nw/iw + dx
        box[:, [1, 3]] = box[:, [1, 3]]*nh/ih + dy
        if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] > w] = w
        box[:, 3][box[:, 3] > h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
        if len(box) > max_boxes: box = box[:max_boxes]
        box_data[:len(box)] = box
    
    return image_data, box_data


def preprocess_true_boxes(true_boxes: np.ndarray, input_shape: tuple,
                          anchors: np.ndarray, num_classes: int):
    assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
    # 一共有三个特征层数
    num_layers = len(anchors)//3
    # ----------------------------------------------------------- #
    #   13x13、26x26特征层对应的anchor均有3个
    # ----------------------------------------------------------- #
    anchor_mask = [[3, 4, 5], [0, 1, 2]]

    # ----------------------------------------------------------- #
    #   获得框的坐标和图片的大小
    # ----------------------------------------------------------- #
    true_boxes = true_boxes.astype('float32')
    input_shape = np.array(input_shape).astype('int32')
    #-----------------------------------------------------------#
    #   通过计算获得真实框的中心和宽高
    #   中心点(m,n,2) 宽高(m,n,2)
    #-----------------------------------------------------------#
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    #-----------------------------------------------------------#
    #   将真实框归一化到小数形式
    #-----------------------------------------------------------#
    true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]

    # m为图片数量，grid_shapes为网格的shape
    m = true_boxes.shape[0]
    grid_shapes = [input_shape//[32, 16][l] for l in range(num_layers)]
    #-----------------------------------------------------------#
    #   y_true的格式为(m,13,13,3,c)(m,26,26,3,c)(m,52,52,3,c)
    #-----------------------------------------------------------#
    y_true = [np.zeros((m, *grid_shapes[l], len(anchor_mask[l]), 5+num_classes),
              dtype='float32') for l in range(num_layers)]

    #-----------------------------------------------------------#
    #   [6,2] -> [1,6,2]
    #-----------------------------------------------------------#
    anchors = anchors[np.newaxis]
    anchor_maxes = anchors

    #-----------------------------------------------------------#
    #   长宽必须大于0
    #-----------------------------------------------------------#
    valid_mask = boxes_wh[..., 0] > 0

    for b in range(m):
        # 对每一张图进行处理
        if not valid_mask[b].any():
            continue
        wh = boxes_wh[b, valid_mask[b]]  # bool数组valid_mask[b]具备数据索引
        #-----------------------------------------------------------#
        #   [n,2] -> [n,1,2]
        #-----------------------------------------------------------#
        wh = wh[:, np.newaxis]
        box_maxes = wh
        #-----------------------------------------------------------#
        #   计算真实框和先验框的交并比
        #   intersect_area  [n,6]
        #   box_area        [n,1]
        #   anchor_area     [1,6]
        #   iou             [n,6]
        #-----------------------------------------------------------#
        intersect_mins = np.zeros_like(box_maxes)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]

        iou = intersect_area / (box_area + anchor_area - intersect_area)
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            #-----------------------------------------------------------#
            #   找到每个真实框所属的特征层
            #-----------------------------------------------------------#
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    #-----------------------------------------------------------#
                    #   floor用于向下取整，找到真实框所属的特征层对应的x、y轴坐标
                    #-----------------------------------------------------------#
                    i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                    #-----------------------------------------------------------#
                    #   k指的的当前这个特征点的第k个先验框
                    #-----------------------------------------------------------#
                    k = anchor_mask[l].index(n)
                    #-----------------------------------------------------------#
                    #   c指的是当前这个真实框的种类
                    #-----------------------------------------------------------#
                    c = true_boxes[b, t, 4].astype('int32')
                    #-----------------------------------------------------------#
                    #   y_true的shape为(m,13,13,3,c)(m,26,26,3,c)
                    #   最后的c可以拆分为框的中心、宽高、置信度以及类别概率
                    #-----------------------------------------------------------#
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5+c] = 1

    return y_true


def cosine_decay_with_warmup(global_step,
                              total_steps,
                              warmup_steps,
                              hold_steps,
                              learning_rate_base,
                              warmup_learning_rate,
                              min_learning_rate):

    if any([learning_rate_base, warmup_learning_rate, min_learning_rate]) < 0:
        raise ValueError('all of the learning rates must be greater than 0.')

    if np.logical_or(total_steps < warmup_steps, total_steps < hold_steps):
        raise ValueError('total_steps must be larger or equal to the other steps.')

    if np.logical_or(learning_rate_base < min_learning_rate, warmup_learning_rate < min_learning_rate):
        raise ValueError('learning_rate_base and warmup_learning_rate must be larger or equal to min_learning_rate.')

    if learning_rate_base < warmup_learning_rate:
        raise ValueError('learning_rate_base must be larger or equal to warmup_learning_rate.')

    if global_step < warmup_steps:

        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        return slope * global_step + warmup_learning_rate

    elif warmup_steps <= global_step <= warmup_steps + hold_steps:

        return learning_rate_base

    else:
        return 0.5 * learning_rate_base * (1 + np.cos(np.pi * (global_step - warmup_steps - hold_steps)/
                                                      (total_steps - warmup_steps - hold_steps)))


class WarmUpCosineDecayScheduler:

    def __init__(self,
                 global_step=0,
                 global_step_init=0,
                 global_interval_steps=None,
                 warmup_interval_steps=None,
                 hold_interval_steps=None,
                 learning_rate_base=None,
                 warmup_learning_rate=None,
                 min_learning_rate=None,
                 interval_epoch=[0.05, 0.15, 0.3, 0.5],
                 verbose=None,
                 **kwargs):
        self.global_step = global_step
        self.global_steps_for_interval = global_step_init
        self.global_interval_steps = global_interval_steps
        self.warmup_interval_steps = warmup_interval_steps
        self.hold_interval_steps = hold_interval_steps
        self.learning_rate_base = learning_rate_base
        self.warmup_learning_rate = warmup_learning_rate
        self.min_learning_rate = min_learning_rate
        self.interval_index = 0
        self.interval_epoch = interval_epoch
        self.interval_reset = [self.interval_epoch[0]]
        for i in range(len(self.interval_epoch) - 1):
            self.interval_reset.append(self.interval_epoch[i+1]-self.interval_epoch[i])
        self.interval_reset.append(1 - self.interval_epoch[-1])
        self.verbose = verbose

    def batch_begin(self):
        if self.global_steps_for_interval in [0] + [int(j * self.global_interval_steps) for j in self.interval_epoch]:
            self.total_steps = int(self.global_interval_steps * self.interval_reset[self.interval_index])
            self.warmup_steps = int(self.warmup_interval_steps * self.interval_reset[self.interval_index])
            self.hold_steps = int(self.hold_interval_steps * self.interval_reset[self.interval_index])
            self.interval_index += 1
            self.global_step = 0

        lr = cosine_decay_with_warmup(global_step=self.global_step,
                                      total_steps=self.total_steps,
                                      warmup_steps=self.warmup_steps,
                                      hold_steps=self.hold_steps,
                                      learning_rate_base=self.learning_rate_base,
                                      warmup_learning_rate=self.warmup_learning_rate,
                                      min_learning_rate=self.min_learning_rate)

        self.global_step += 1
        self.global_steps_for_interval += 1

        if self.verbose:
            print('\nBatch %05d: setting learning '
                  'rate to %s.' % (self.global_steps_for_interval, lr))

        return lr


def calculate_score(y_trues, y_preds, object_masks: list, depth: int):

    scores = []
    for y_true, y_pred, object_mask in zip(y_trues, y_preds, object_masks):

        if object_mask.any():
            y_true = y_true[..., 5:][object_mask]
            y_pred = y_pred[..., 5:][object_mask]
            true_class = y_true.argmax(dim=-1)
            pred_class = y_pred.argmax(dim=-1)
            for i in range(depth):
                precision_num = torch.eq(pred_class, i).float().sum()
                if precision_num:

                    true_bool_mask = torch.eq(true_class, i)
                    pred_bool_mask = torch.eq(pred_class, i)

                    bool_mask = torch.stack([true_bool_mask, pred_bool_mask], dim=-1).all(dim=-1)
                    true_positive_num = bool_mask.float().sum()

                else:
                    continue

                false_negative_num = true_bool_mask.float().sum() - true_positive_num
                recall_num = true_positive_num + false_negative_num

                if recall_num:

                    precision = true_positive_num / precision_num
                    recall = true_positive_num / recall_num

                else:
                    continue

                score = 2 * (precision * recall) / (precision + recall)
                scores.append(score)

    f1_score = torch.square(torch.mean(torch.tensor(scores)))

    return torch.zeros(size=()) if torch.isnan(f1_score) else f1_score
