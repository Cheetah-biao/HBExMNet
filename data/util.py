import random

import numpy as np


def augment(img_list, hflip=True, rot=True):
    use_hflip = hflip and random.random() < 0.5
    use_vflip = rot and random.random() < 0.5
    use_dflip = rot and random.random() < 0.5
    use_transpose = rot and random.random() < 0.5

    def _apply(img):
        if use_hflip:
            img = np.flip(img, axis=-1)
        if use_vflip:
            img = np.flip(img, axis=-2)
        if img.ndim >= 5 and use_dflip:
            img = np.flip(img, axis=-3)
        if img.ndim >= 5 and use_transpose:
            img = np.swapaxes(img, -1, -2)
        return np.ascontiguousarray(img)

    return [_apply(img) for img in img_list]
