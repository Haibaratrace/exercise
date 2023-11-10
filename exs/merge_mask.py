from PIL import Image
import numpy as np


def merge_mask(img, mask):
    img_array = np.array(Image.open(img))
    mask_array = np.array(Image.open(mask))

    new_img = np.concatenate((img_array, mask_array[:, :, 0:1]), axis=-1)
    img = Image.fromarray(new_img.astype('unit8'), 'RGBA')
    return img
