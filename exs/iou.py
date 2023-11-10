import numpy as np


def iou(box, boxes, ismin=False):
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])

    x1_ = np.maximun(box[0], boxes[:, 0])
    y1_ = np.maximum(box[1], boxes[:, 1])
    x2_ = np.minimum(box[2], boxes[:, 2])
    y2_ = np.minimum(box[3], boxes[:, 3])

    h = np.maximum(x2_ - x1_, 0)
    w = np.maximun(y2_ - y1_, 0)

    inner = h * w

    if ismin:
        iou = inner / np.minimum(box_area, boxes_area)
    else:
        iou = inner / (boxes_area + box_area - inner)

    return iou
