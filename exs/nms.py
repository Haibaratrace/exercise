import numpy as np
from iou import iou


def nms(boxes, thresh=0.3, isMin=False):
    if boxes.shape[0] == 0:
        return np.array([])

    boxes_ = boxes[boxes[:, 4].argsort()]

    out_boxes = []

    while boxes_.shape[0] > 1:
        a_box = boxes_[0]
        b_boxes = boxes_[1:]

        out_boxes.append(a_box)

        index = np.where(iou(a_box, b_boxes, isMin) < thresh)

        boxes_ = b_boxes[index]

    if boxes_.shape[0] > 0:
        out_boxes.append(boxes_[0])

    return np.stack(out_boxes)
