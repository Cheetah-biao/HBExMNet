import os
import time
import numpy as np
import torch

def _raise(e):
    raise e

def normalize_min_max(im):
    eps = 1e-10
    min_v = np.min(im)
    max_v = np.max(im)
    im = (im - min_v) / (max_v - min_v + eps)
    return im.astype(np.float32)

class Predictor:
    def __init__(self,
                 test_model,
                 factor=6,
                 block_size=None,
                 dtype=np.float32):
        if block_size is None:
            block_size = [20, 50, 50]
        self.factor = factor
        self.block_size = block_size
        self.test_model = test_model
        self.dtype = dtype

    def __predict_block(self, block):
        return self.test_model(block)

    def __check_dims(self, im, block_size, overlap):
        # To-Do: deal with float block_size and overlap
        def __broadcast(val):
            val = [val for i in im.shape] if np.isscalar(val) else val
            return val

        len(im.shape) == 3 or len(im.shape) == 2 or _raise(
            ValueError('the input image must be in shape [depth, height, width]'))

        block_size = __broadcast(block_size)
        overlap = __broadcast(overlap)

        len(block_size) == len(im.shape) or _raise(
            ValueError("ndim of block_size ({}) mismatch that of image size ({})".format(block_size, im.shape)))
        len(overlap) == len(im.shape) or _raise(
            ValueError("ndim of overlap ({}) mismatch that of image size ({})".format(overlap, im.shape)))

        # block_size = [b if b <= i else i for b, i in zip(block_size, im.shape)]

        overlap = [i if i > 1 else i * s for i, s in zip(overlap, block_size)]
        overlap = [0 if b >= s else i for i, b, s in zip(overlap, block_size,
                                                         im.shape)]  # no overlap along the dims where the image size equal to the block size
        overlap = [i if i % 2 == 0 else i + 1 for i in overlap]  # overlap must be even number

        block_size = [b - 2 * i for b, i in zip(block_size, overlap)]  # real block size when inference

        overlap = [int(i) for i in overlap]
        block_size = [int(i) for i in block_size]
        print('block size (overlap excluded) : {} overlap : {}'.format(block_size, overlap))

        return block_size, overlap

    def _padding_block(self, im, blk_size, overlap):
        grid_dim = [int(np.ceil(float(i) / b)) for i, o, b in zip(im.shape, overlap, blk_size)]
        im_size_padded = [(g * b + b if o != 0 else g * b) for g, b, o in zip(grid_dim, blk_size, overlap)]

        im_wrapped = np.ones(im_size_padded, dtype=self.dtype) * np.min(im)

        valid_region = [slice(o // 2, o // 2 + i) for o, i in zip(overlap, im.shape)]
        sr_valid_region = [slice(o // 2 * self.factor, (o // 2 + i) * self.factor) for o, i in zip(overlap, im.shape)]
        print('raw image size : {}, wrapped into : {}'.format(im.shape, im_size_padded))
        print('valid region index: {} ({} after SR)'.format(valid_region, sr_valid_region))

        im_wrapped[tuple(valid_region)] = im

        return im_wrapped, sr_valid_region

    def __region_iter(self, im, blk_size, overlap, factor):
        """
        Params:
            -im: ndarray in dims of [depth, height, width]
        """
        im_size = im.shape

        anchors = [(z, y, x)
                   for z in range(overlap[0], im_size[0], blk_size[0])
                   for y in range(overlap[1], im_size[1], blk_size[1])
                   for x in range(overlap[2], im_size[2], blk_size[2])]

        for i, anchor in enumerate(anchors):
            # revised_overlap = [0 if a == i else i for a, i in zip(anchor, overlap)]
            begin = [p - c for p, c in zip(anchor, overlap)]
            end = [p + b + c for p, b, c in zip(anchor, blk_size, overlap)]
            yield [slice(b, e) for b, e in zip(begin, end)], \
                  [slice((b + c // 2) * factor, (e - c // 2) * factor) for b, e, c, in zip(begin, end, overlap)], \
                  [slice((c // 2) * factor, (b + c + c // 2) * factor) for b, c in zip(blk_size, overlap)]

    def predict_without_norm(self, im, block_size, overlap):
        block_size, overlap = self.__check_dims(im, block_size, overlap)
        factor = self.factor

        im_wrapped, valid_region_idx = self._padding_block(im, block_size, overlap)
        sr_size = [s * factor for s in im_wrapped.shape]
        sr = np.zeros(sr_size, dtype=self.dtype)

        for src, dst, in_blk in self.__region_iter(im_wrapped, block_size, overlap, factor):
            # print('source: {}  dst: {}  valid: {} '.format(src, dst, in_blk))
            begin = [i.start for i in src]
            end = [i.stop for i in src]

            if not all(i <= j for i, j in zip(end, im_wrapped.shape)):
                continue

            print('\revaluating {}-{} in {}  '.format(begin, end, im_wrapped.shape), end='')
            block = im_wrapped[tuple(src)]
            block = self.__predict_block(block)
            sr[tuple(dst)] = block[tuple(in_blk)]

        print('')
        return sr[tuple(valid_region_idx)]


    def predict(self, im, block_size, overlap, normalization='fixed', **kwargs):
        print('normalized to [%.4f, %.4f]' % (np.min(im), np.max(im)))
        im = torch.from_numpy(
            np.ascontiguousarray(np.transpose(normalize_min_max(im), (3, 0, 1, 2)))).float()
        sr = self.predict_without_norm(im, block_size, overlap)
        return sr

    def predict_iso(self, im, block_size, overlap, normalization='fixed', **kwargs):
        sr = self.predict_without_norm(im, block_size, overlap)
        return sr

