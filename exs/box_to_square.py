import numpy as np


def convert_to_square(bbox):
    square_box = bbox.copy()
    if bbox.shape[0] == 0:
        return np.array([])
    h = bbox[:, 2] - bbox[:, 0]
    w = bbox[:, 3] - bbox[:, 1]

    s_len = np.maximum(h, w)

    square_box[:, 0] = bbox[:, 0] + h / 2 - s_len / 2
    square_box[:, 1] = bbox[:, 1] + w / 2 - s_len / 2
    square_box[:, 2] = square_box[:, 0] + s_len
    square_box[:, 3] = square_box[:, 1] + s_len

    return square_box
