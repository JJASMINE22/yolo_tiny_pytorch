import torch
import torchvision
from torch import nn
from _utils import cfg

def box_iou(pred_boxes, true_boxes):
    # 13,13,3,1,4
    # 计算左上角的坐标和右下角的坐标
    pred_boxes = pred_boxes.unsqueeze(dim=-2)
    pred_boxes_xy = pred_boxes[..., :2]
    pred_boxes_wh = pred_boxes[..., 2:4]
    pred_boxes_wh_half = pred_boxes_wh/2.
    pred_boxes_mins = pred_boxes_xy - pred_boxes_wh_half
    pred_boxes_maxes = pred_boxes_xy + pred_boxes_wh_half

    # 1,n,4
    # 计算左上角和右下角的坐标
    true_boxes = true_boxes.unsqueeze(dim=0)
    true_boxes_xy = true_boxes[..., :2]
    true_boxes_wh = true_boxes[..., 2:4]
    true_boxes_wh_half = true_boxes_wh/2.
    true_boxes_mins = true_boxes_xy - true_boxes_wh_half
    true_boxes_maxes = true_boxes_xy + true_boxes_wh_half

    # 通过拓展域, 计算重叠面积, pred:(13, 13, 3) → 1, 1 → n
    intersect_mins = torch.maximum(pred_boxes_mins, true_boxes_mins)
    intersect_maxes = torch.minimum(pred_boxes_maxes, true_boxes_maxes)
    intersect_wh = torch.maximum(intersect_maxes - intersect_mins, torch.zeros(size=()))
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    pred_boxes_area = pred_boxes_wh[..., 0] * pred_boxes_wh[..., 1]
    true_boxes_area = true_boxes_wh[..., 0] * true_boxes_wh[..., 1]
    iou = intersect_area / (pred_boxes_area + true_boxes_area - intersect_area)

    return iou

def get_ignore_mask(real_boxes, pred_boxes, object_mask, ignore_threshold):

    ignore_masks = []
    batch_size = real_boxes.size(0)
    for i in range(batch_size):
        iou = box_iou(pred_boxes[i, ..., :4], real_boxes[i][object_mask[i]])
        if iou.size(-1):
            iou_mask = torch.less(torch.max(iou, dim=-1)[0], ignore_threshold)
            ignore_masks.append(iou_mask)
        else:
            ignore_masks.append(torch.zeros(size=iou.size()[:-1], device=cfg.device).bool())
    ignore_mask = torch.stack(ignore_masks, dim=0)

    return ignore_mask


def smooth_labels(target, smoothing_rate):

    num_classes = target.size(-1)
    smoothed_target = target * (1. - smoothing_rate) + smoothing_rate / num_classes

    return smoothed_target


def yolo_head(feats, anchors, num_classes, input_shape, training: bool=True):
    num_anchors = len(anchors)
    #---------------------------------------------------#
    #   [1, 1, 1, num_anchors, 2]
    #---------------------------------------------------#
    anchors = anchors.view(1, 1, 1, num_anchors, 2)

    #---------------------------------------------------#
    #   获得x，y的网格
    #   (13, 13, 1, 2)
    #---------------------------------------------------#
    grid_shape = feats.size()[1:3]
    grids = torch.meshgrid(torch.arange(grid_shape[0]),
                           torch.arange(grid_shape[1]), indexing='xy')
    grids = torch.stack(grids, axis=-1)
    grids = grids.unsqueeze(dim=-2).to(cfg.device) if cfg.device else grids.unsqueeze(dim=-2)

    #---------------------------------------------------#
    #   将预测结果调整成(batch_size,13,13,3,85)
    #   85可拆分成4 + 1 + 80
    #   4代表的是中心宽高的调整参数
    #   1代表的是框的置信度
    #   80代表的是种类的置信度
    #---------------------------------------------------#
    feats = feats.view(-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5)

    #---------------------------------------------------#
    #   将预测值调成真实值
    #   box_xy对应框的中心点
    #   box_wh对应框的宽和高
    #---------------------------------------------------#
    box_xy = (torch.sigmoid(feats[..., :2]) + grids.float().unsqueeze(dim=0)) / \
             torch.tensor(grid_shape, dtype=torch.float).to(cfg.device) \
        if cfg.device else grids.unsqueeze(dim=-2)
    box_wh = torch.exp(feats[..., 2:4]) * anchors / input_shape

    #---------------------------------------------------------------------#
    #   在计算loss的时候返回grid, feats, box_xy, box_wh
    #   在预测的时候返回box_xy, box_wh, box_confidence, box_class_probs
    #---------------------------------------------------------------------#

    box_confidence = torch.sigmoid(feats[..., 4:5])
    box_class_probs = feats[..., 5:]
    if training:
        box_class_probs = torch.sigmoid(box_class_probs)

    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    # -----------------------------------------------------------------#
    #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
    # -----------------------------------------------------------------#
    box_yx = box_xy.flip(dims=[-1]) # box_xy[..., ::-1]
    box_hw = box_wh.flip(dims=[-1]) # box_wh[..., ::-1]

    new_shape = torch.round(image_shape * torch.min(input_shape/image_shape))
    # -----------------------------------------------------------------#
    #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
    #   new_shape指的是宽高缩放情况
    # -----------------------------------------------------------------#
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape

    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = torch.cat([
        box_mins[..., 0:1] * image_shape[0],  # y_min
        box_mins[..., 1:2] * image_shape[1],  # x_min
        box_maxes[..., 0:1] * image_shape[0],  # y_max
        box_maxes[..., 1:2] * image_shape[1]  # x_max
    ], axis=-1)

    return boxes


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape, letterbox: bool=False):
    #-----------------------------------------------------------------#
    #   将预测值调成真实值
    #   box_xy : -1,13,13,3,2;
    #   box_wh : -1,13,13,3,2;
    #   box_confidence : -1,13,13,3,1;
    #   box_class_probs : -1,13,13,3,80;
    #-----------------------------------------------------------------#
    box_xy, box_wh, box_confidences, box_class_probs = yolo_head(feats, anchors, num_classes,
                                                                 input_shape, training=False)
    #-----------------------------------------------------------------#
    #   在图像传入网络预测前会进行letterbox_image给图像周围添加灰条
    #   因此生成的box_xy, box_wh是相对于有灰条的图像的
    #   我们需要对齐进行修改，去除灰条的部分。
    #   将box_xy、和box_wh调节成y_min,y_max,xmin,xmax
    #-----------------------------------------------------------------#
    if letterbox:
        boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    else:
        # reverse x and y axes
        box_yx = box_xy.flip(dims=[-1]) # box_xy[..., ::-1]
        box_hw = box_wh.flip(dims=[-1]) # box_wh[..., ::-1]
        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)

        boxes = torch.cat([
            box_mins[..., 0:1] * image_shape[0],  # y_min
            box_mins[..., 1:2] * image_shape[1],  # x_min
            box_maxes[..., 0:1] * image_shape[0],  # y_max
            box_maxes[..., 1:2] * image_shape[1]  # x_max
        ], dim=-1)
    #-----------------------------------------------------------------#
    #   获得最终得分和框的位置
    #-----------------------------------------------------------------#
    boxes = boxes.view(-1, 4)
    box_confidences = box_confidences.reshape(-1)
    box_class_probs = box_class_probs.reshape(-1, num_classes)

    return boxes, box_confidences, box_class_probs


def yolo_eval(anchors,
              num_classes,
              image_shape,
              yolo_outputs,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5,
              letterbox_image=False):
    # ---------------------------------------------------#
    #   获得特征层的数量，有效特征层的数量为3
    # ---------------------------------------------------#
    num_layers = len(yolo_outputs)
    # -----------------------------------------------------------#
    #   13x13的特征层对应的anchor是[142, 110], [192, 243], [459, 401]
    #   26x26的特征层对应的anchor是[36, 75], [76, 55], [72, 146]
    #   52x52的特征层对应的anchor是[12, 16], [19, 36], [40, 28]
    # -----------------------------------------------------------#
    anchor_mask = [[3, 4, 5], [0, 1, 2]]

    # -----------------------------------------------------------#
    #   这里获得的是输入图片的大小，一般是416x416
    # -----------------------------------------------------------#
    batch_size = yolo_outputs[0].size(0)
    input_shape = torch.tensor(yolo_outputs[0].size()[1:3],
                               dtype=torch.float, device=cfg.device) * 32
    total_boxes = list()
    total_scores = list()
    total_classes = list()
    for i in range(batch_size):
        # -----------------------------------------------------------#
        #   遍历批量
        # -----------------------------------------------------------#
        boxes = list()
        box_scores = list()
        box_class_probs = list()
        for l in range(num_layers):
            # -----------------------------------------------------------#
            #   对每个特征层进行处理
            # -----------------------------------------------------------#
            _boxes, _box_confs, _box_class_probs = yolo_boxes_and_scores(yolo_outputs[l][i].unsqueeze(dim=0),
                                                                         anchors[anchor_mask[l]], num_classes,
                                                                         input_shape, image_shape,
                                                                         letterbox_image)
            boxes.append(_boxes)
            box_scores.append(_box_confs)
            box_class_probs.append(_box_class_probs)
        # -----------------------------------------------------------#
        #   将每个特征层的结果进行堆叠
        # -----------------------------------------------------------#
        boxes = torch.cat(boxes, dim=0)
        box_scores = torch.cat(box_scores, axis=0)
        box_class_probs = torch.cat(box_class_probs, axis=0)

        # -----------------------------------------------------------#
        #   判断得分是否大于score_threshold
        # -----------------------------------------------------------#
        mask = box_scores >= score_threshold
        masked_scores = box_scores[mask]
        masked_boxes = boxes[mask]
        masked_probs = box_class_probs[mask]
        boxes_ = list()
        scores_ = list()
        classes_ = list()
        for c in range(num_classes):
            # -----------------------------------------------------------#
            #   取出所有box_scores >= score_threshold的框
            # -----------------------------------------------------------#
            class_boxes = masked_boxes[torch.eq(masked_probs.argmax(dim=-1), c)]
            class_box_scores = masked_scores[torch.eq(masked_probs.argmax(dim=-1), c)]

            # -----------------------------------------------------------#
            #   非极大抑制
            # -----------------------------------------------------------#

            nms_index = torchvision.ops.nms(class_boxes, class_box_scores, iou_threshold)
            nms_index = nms_index[:max_boxes]
            # -----------------------------------------------------------#
            #   框的位置，得分与种类
            # -----------------------------------------------------------#
            class_boxes = class_boxes[nms_index]
            class_box_scores = class_box_scores[nms_index]
            classes = torch.ones_like(class_box_scores, dtype=torch.int32) * c
            boxes_.append(class_boxes)
            scores_.append(class_box_scores)
            classes_.append(classes)
        boxes_ = torch.cat(boxes_, axis=0)
        scores_ = torch.cat(scores_, axis=0)
        classes_ = torch.cat(classes_, axis=0)
        total_boxes.append(boxes_)
        total_scores.append(scores_)
        total_classes.append(classes_)

    return total_boxes, total_scores, total_classes
