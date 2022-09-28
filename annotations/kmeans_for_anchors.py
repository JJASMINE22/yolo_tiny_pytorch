import cv2
import glob
import numpy as np
import xml.etree.ElementTree as ET
from annotations import cfg


def cas_iou(box,cluster):
    x = np.minimum(cluster[:,0],box[0])
    y = np.minimum(cluster[:,1],box[1])

    intersection = x * y
    area1 = box[0] * box[1]

    area2 = cluster[:,0] * cluster[:,1]
    iou = intersection / (area1 + area2 - intersection)

    return iou

def avg_iou(box,cluster):
    return np.mean([np.max(cas_iou(box[i],cluster)) for i in range(box.shape[0])])


def kmeans(box, k):
    """
    :param box: actual detection frames
    :param k: number of a priori frames (num of sensory fields × num of sizes)
    :return:
    """
    row = box.shape[0]
    # initialize the (1-iou) of all frames corresponding to priori frames
    distance = np.empty((row, k))
    # initialize the optimal (1-iou) of all frames
    last_clu = np.zeros((row,))

    np.random.seed()
    cluster = box[np.random.choice(row, k, replace=False)]
    while True:
        for i in range(row):
            # use iou to replace euclidean distance
            distance[i] = 1 - cas_iou(box[i], cluster)

        near = np.argmin(distance, axis=1)
        # loop termination condition
        if (last_clu == near).all():
            break

        # update
        for j in range(k):
            cluster[j] = np.median(
                box[near == j],axis=0)

        last_clu = near

    return cluster

def load_data(path):
    tree = ET.parse(path)
    height = int(tree.findtext('./size/height'))
    width = int(tree.findtext('./size/width'))
    if height <= 0 or width <= 0:
        return None

    # get coordinates of each obj
    for obj in tree.iter('object'):
        # 比例
        xmin = int(float(obj.findtext('bndbox/xmin'))) / width
        ymin = int(float(obj.findtext('bndbox/ymin'))) / height
        xmax = int(float(obj.findtext('bndbox/xmax'))) / width
        ymax = int(float(obj.findtext('bndbox/ymax'))) / height

        xmin = np.float64(xmin)
        ymin = np.float64(ymin)
        xmax = np.float64(xmax)
        ymax = np.float64(ymax)

    # get height&width
    return [xmax - xmin, ymax - ymin]

def read_data(line):
    line = line.split()
    h, w, _ = cv2.imread(line[0]).shape
    boxes = np.array([list(map(lambda x: int(x), coordinate.split(',')[:-1]))
                      for coordinate in line[1:]], dtype='float')
    boxes[:, [0, 2]] /= w
    boxes[:, [1, 3]] /= h

    return boxes[:, [2, 3]] - boxes[:, [0, 1]]


if __name__ == '__main__':

    # generate yolo_anchors.txt
    SIZE = cfg.input_size
    anchors_num = 6  # cfg.anchors.__len__()  # number of a priori frames (sensory fields × sizes)

    f = open(cfg.annotation_path, 'r')
    lines = f.readlines()
    data = list()
    for line in lines:
        wh = read_data(line)[0]
        data.append(wh)

    data = np.array(data)

    # k-means
    out = kmeans(data, anchors_num)
    out = out[np.argsort(out[:,0])]
    print('acc:{:.2f}%'.format(avg_iou(data, out) * 100))
    print('anchor_x: {}, anchor_y: {}'.format(out*SIZE[1], out*SIZE[0]))
    data = out*list(reversed(SIZE))
    f = open(cfg.anchors_path, 'w')

    for i in range(np.shape(data)[0]):
        if i == 0:
            x_y = "%d,%d" % (data[i][0], data[i][1])
        else:
            x_y = ", %d,%d" % (data[i][0], data[i][1])
        f.write(x_y)
    f.close()
