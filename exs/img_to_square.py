import PIL.Image as pimg


def img_to_square(img, scale_side):
    img.thumbnail((scale_side, scale_side))
    w2, h2 = img.size
    bg_img = pimg.new('RGB', (scale_side, scale_side))

    if w2 == scale_side:
        bg_img.paste(img, (0, int((scale_side - h2) / 2)))
    elif h2 == scale_side:
        bg_img.paste(img, (int((scale_side - w2) / 2), 0))
    else:
        bg_img.paste(img, (int((scale_side - w2) / 2), int((scale_side - h2) / 2)))

    return bg_img
