import math
import torch
from torch import nn
from torch.nn import functional as F
from _utils.yolo_utils import yolo_head, smooth_labels, get_ignore_mask
from custom import cfg

def box_ciou(pred_boxes, true_boxes):
    """
    :param pred_boxes: shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    :param true_boxes: shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    :return: ciou, shape=(batch, feat_w, feat_h, anchor_num, 1)
    """
    # -----------------------------------------------------------#
    #   求出预测框左上角右下角
    #   pred_boxes_mins     (batch, feat_w, feat_h, anchor_num, 2)
    #   pred_boxes_maxes    (batch, feat_w, feat_h, anchor_num, 2)
    # -----------------------------------------------------------#
    pred_boxes_xy = pred_boxes[..., :2]
    pred_boxes_wh = pred_boxes[..., 2:4]
    pred_boxes_wh_half = pred_boxes_wh / 2.
    pred_boxes_mins = pred_boxes_xy - pred_boxes_wh_half
    pred_boxes_maxes = pred_boxes_xy + pred_boxes_wh_half
    # -----------------------------------------------------------#
    #   求出真实框左上角右下角
    #   true_boxes_mins     (batch, feat_w, feat_h, anchor_num, 2)
    #   true_boxes_maxes    (batch, feat_w, feat_h, anchor_num, 2)
    # -----------------------------------------------------------#
    true_boxes_xy = true_boxes[..., :2]
    true_boxes_wh = true_boxes[..., 2:4]
    true_boxes_wh_half = true_boxes_wh / 2.
    true_boxes_mins = true_boxes_xy - true_boxes_wh_half
    true_boxes_maxes = true_boxes_xy + true_boxes_wh_half

    # -----------------------------------------------------------#
    #   求真实框和预测框所有的iou
    #   iou         (batch, feat_w, feat_h, anchor_num)
    # -----------------------------------------------------------#
    intersect_mins = torch.maximum(pred_boxes_mins, true_boxes_mins)
    intersect_maxes = torch.minimum(pred_boxes_maxes, true_boxes_maxes)
    intersect_wh = torch.maximum(intersect_maxes - intersect_mins, torch.zeros(size=()))
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    pred_boxes_area = pred_boxes_wh[..., 0] * pred_boxes_wh[..., 1]
    true_boxes_area = true_boxes_wh[..., 0] * true_boxes_wh[..., 1]
    union_area = pred_boxes_area + true_boxes_area - intersect_area
    iou = intersect_area / torch.maximum(union_area, torch.tensor(1e-7))  # 防止nan

    # -----------------------------------------------------------#
    #   计算中心的差距
    #   center_distance (batch, feat_w, feat_h, anchor_num)
    # -----------------------------------------------------------#
    center_distance = torch.square(pred_boxes_xy - true_boxes_xy).sum(dim=-1)
    enclose_mins = torch.minimum(pred_boxes_mins, true_boxes_mins)
    enclose_maxes = torch.maximum(pred_boxes_maxes, true_boxes_maxes)
    enclose_wh = torch.maximum(enclose_maxes - enclose_mins, torch.zeros(size=()))
    # -----------------------------------------------------------#
    #   计算对角线距离
    #   enclose_diagonal (batch, feat_w, feat_h, anchor_num)
    # -----------------------------------------------------------#
    enclose_diagonal = torch.square(enclose_wh).sum(dim=-1)
    ciou = iou - 1.0 * (center_distance) / torch.maximum(enclose_diagonal, torch.tensor(1e-7))  # 防止nan

    v = 4 * torch.square(torch.atan2(pred_boxes_wh[..., 0], torch.maximum(pred_boxes_wh[..., 1], torch.tensor(1e-7))) -
                         torch.atan2(true_boxes_wh[..., 0], torch.maximum(true_boxes_wh[..., 1], torch.tensor(1e-7)))) / (math.pi ** 2)
    alpha = v / torch.maximum((1.0 - iou + v), torch.tensor(1e-7))  # 防止当形状与y_true一致时, v为0的情况
    ciou = ciou - alpha * v

    ciou = ciou.unsqueeze(dim=-1)
    ciou = torch.where(torch.isnan(ciou), torch.zeros_like(ciou), ciou)
    return ciou


class MaskedCrossEntropy(nn.Module):
    """
    calculate yolo classification error based on NLL
    by using object mask
    """
    def __init__(self,
                 **kwargs):
        super(MaskedCrossEntropy, self).__init__(**kwargs)
        self.bce_loss = nn.BCELoss(reduction='none')

    def forward(self, y_pred, y_true, mask: torch.Tensor):
        """
        in eager execution,
        do not use tf's logical operators、iterators
        """
        masked_true_probs = y_true[mask]
        masked_pred_probs = y_pred[mask]

        target_size = masked_true_probs.size(0)
        class_num = masked_true_probs.size(-1)
        if not target_size:
            return 0.

        total_loss = []

        for i in range(class_num):

            masked_true_prob = masked_true_probs[torch.gt(masked_true_probs[:, i], .5)]
            masked_pred_prob = masked_pred_probs[torch.gt(masked_true_probs[:, i], .5)]
            if masked_true_prob.size(0) >= target_size // 2 and target_size >= 1:
                loss = self.bce_loss(masked_pred_prob, masked_true_prob).sum(dim=-1)
                total_loss.append(torch.mean(torch.topk(loss, k=target_size // 2)[0]))
            elif masked_true_prob.size(0) == 0:
                pass
            else:
                loss = self.bce_loss(masked_pred_prob, masked_true_prob).sum(dim=-1)
                total_loss.append(torch.mean(loss))

        total_loss = torch.stack(total_loss)

        return torch.mean(total_loss)

masked_bce_loss = MaskedCrossEntropy()
bce_loss = torch.nn.BCELoss(reduction='none')

def yolo_loss(y_true, y_pred, anchors, num_classes, ignore_thresh=.5, label_smoothing=0.1):
    # 一共有两层
    num_layers = len(anchors) // 3

    # -----------------------------------------------------------#
    #   13x13的特征层对应的anchor是[142, 110], [192, 243], [459, 401]
    #   26x26的特征层对应的anchor是[36, 75], [76, 55], [72, 146]
    #   52x52的特征层对应的anchor是[12, 16], [19, 36], [40, 28]
    # -----------------------------------------------------------#
    anchor_mask = [[3, 4, 5], [1, 2, 3]] if num_layers == 2 else [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    # 得到input_shape, batch_size
    input_shape = torch.tensor(y_pred[0].size()[1:3],
                               dtype=torch.float, device=cfg.device) * 32
    batch_size = y_true[0].size(0)

    # 初始化2个感受野下的总误差
    loss = 0

    for l in range(num_layers):
        # -----------------------------------------------------------#
        #   以第一个特征层(m,13,13,3,85)为例子
        #   取出该特征层中存在目标的点的位置。(m,13,13,3,1)
        # -----------------------------------------------------------#
        object_mask = y_true[l][..., 4:5]
        # -----------------------------------------------------------#
        #   取出其对应的种类(m,13,13,3,class_num)
        # -----------------------------------------------------------#
        true_class_probs = y_true[l][..., 5:]
        if label_smoothing:
            true_class_probs = smooth_labels(true_class_probs, label_smoothing)

        # -----------------------------------------------------------#
        #   grid        (13,13,1,2) 网格坐标
        #   raw_pred    (m,13,13,3,5+class_num) 尚未处理的预测结果
        #   pred_xy     (m,13,13,3,2) 解码后的中心坐标
        #   pred_wh     (m,13,13,3,2) 解码后的宽高坐标
        # -----------------------------------------------------------#
        pred_xy, pred_wh, box_confidence, box_class_probs = yolo_head(y_pred[l],
                                                                      anchors[anchor_mask[l]],
                                                                      num_classes, input_shape)

        # -----------------------------------------------------------#
        #   pred_box是解码后的预测的box的位置
        #   (m,13,13,3,4)
        # -----------------------------------------------------------#
        pred_boxes = torch.cat([pred_xy, pred_wh], axis=-1)

        # -----------------------------------------------------------#
        #   找到负样本群组
        #   ignore_mask用于提取出作为负样本的特征点
        #   (m,13,13,3)
        # -----------------------------------------------------------#
        object_bool_mask = object_mask.bool().squeeze(dim=-1)
        ignore_bool_mask = get_ignore_mask(y_true[l][..., :4], pred_boxes, object_bool_mask, ignore_thresh)

        # -----------------------------------------------------------#
        #   真实框越大，比重越小，小框的比重更大。
        # -----------------------------------------------------------#
        box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]

        # -----------------------------------------------------------#
        #   计算Ciou loss
        # -----------------------------------------------------------#
        ciou = box_ciou(pred_boxes, y_true[l][..., 0:4])
        ciou_loss = box_loss_scale * (1 - ciou)
        ciou_loss = ciou_loss[object_bool_mask]

        # ------------------------------------------------------------------------------#
        #   如果该位置本来有框，那么计算1与置信度的交叉熵
        #   如果该位置本来没有框，那么计算0与置信度的交叉熵
        #   在这其中会忽略一部分样本，这些被忽略的样本满足条件best_iou<ignore_thresh
        #   该操作的目的是：
        #   忽略预测结果与真实框非常对应特征点，因为这些框已经比较准了
        #   不适合当作负样本，所以忽略掉。
        # ------------------------------------------------------------------------------#

        pos_conf_loss = bce_loss(box_confidence, object_mask)[object_bool_mask]
        neg_conf_loss = (1 - object_mask) * bce_loss(box_confidence, object_mask)
        neg_conf_loss = neg_conf_loss[ignore_bool_mask]

        # class_loss = bce_loss(true_class_probs, box_class_probs)[object_bool_mask].mean(dim=-1)
        class_loss = masked_bce_loss(box_class_probs, true_class_probs,
                                     mask=object_bool_mask)

        pos_num = torch.maximum(object_mask.sum(), torch.ones(size=()))

        location_loss = ciou_loss.sum()
        pos_conf_loss = pos_conf_loss.sum()
        neg_conf_loss = neg_conf_loss.sum()
        # class_loss = class_loss.sum()

        loss += (location_loss + pos_conf_loss + neg_conf_loss) / pos_num + class_loss

    loss = loss / num_layers

    return loss
