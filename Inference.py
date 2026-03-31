import os.path
import sys
import time
import argparse

import torch
from utils.logger import get_logger
from utils.util import list_any_in_str, slice2str
from utils.files import mkdirs
from options import options
from data import create_dataset, create_dataloader
from models import create_model
import numpy as np
import threading
import tifffile
import imageio
from scipy.ndimage import zoom

logger = get_logger("inference_single_stage")


def _build_inference_selection(args):
    selection = {}
    if args.mode:
        selection["mode"] = args.mode
    if args.organelle:
        selection["organelle"] = args.organelle
    if args.input_path:
        selection["input_path"] = args.input_path
    if args.xy_nm is not None:
        selection["xy_nm"] = args.xy_nm
    if args.z_nm is not None:
        selection["z_nm"] = args.z_nm
    if args.z_zoom is not None:
        selection["z_zoom"] = args.z_zoom
    if args.task == "denoise":
        selection["denoise"] = True
        selection["sr"] = False
    elif args.task == "sr":
        selection["denoise"] = False
        selection["sr"] = True
    elif args.task == "both":
        selection["denoise"] = True
        selection["sr"] = True
    return selection or None


def predict_tile(model, x, net_axes_in_div_by, block_size, overlap, method="cover"):
    def __broadcast(val):
        val = [val for i in x.shape] if np.isscalar(val) else val
        return val

    def _split(v):
        a = v // 2
        return a, v - a

    def slice_generator(x_pad, block_img_size, overlap, factor=1):
        in_slice_list, out_slice_list, crop_slice_list = [], [], []
        anchors = [(z, y, x)
                   for z in range(overlap[0], x_pad.shape[0], block_img_size[0])
                   for y in range(overlap[1], x_pad.shape[1], block_img_size[1])
                   for x in range(overlap[2], x_pad.shape[2], block_img_size[2])]

        for i, anchor in enumerate(anchors):
            begin = [p - o for p, o in zip(anchor, overlap)]  # ensure starting from 0 and with overlap
            end = [p + b + o for p, b, o in zip(anchor, block_img_size, overlap)]
            if not all(i <= j for i, j in zip(end, x_pad.shape)):
                continue

            in_slice_list.append(tuple([slice(b, e) for b, e in zip(begin, end)]))
            # crop and out only take the middle part, discard useless padding and edge pixels
            out_slice_list.append(
                tuple([slice((b + o // 2) * factor, (e - o // 2) * factor) for b, e, o, in zip(begin, end, overlap)]))
            crop_slice_list.append(
                tuple([slice((c // 2) * factor, (b + c + c // 2) * factor) for b, c in zip(block_img_size, overlap)]))

        # for i in range(len(in_slice_list)):
        #     logger.info(f"in slice list: {in_slice_list[i]}, out slice list: {out_slice_list[i]}, crop slice list: {crop_slice_list[i]}")

        return in_slice_list, out_slice_list, crop_slice_list

    assert method in ["mean", "cover"], "invalid method"

    axes = list(range(x.ndim))
    block_size = __broadcast(block_size)
    overlap = __broadcast(overlap)
    net_axes_in_div_by = __broadcast(net_axes_in_div_by)

    # Add 2 * overlap for subsequent cropping to avoid edge pixels and artifacts
    block_img_size = [r // b + 2 * o if b > 1 else r for r, b, o in zip(x.shape, block_size, overlap)]
    block_img_size = [int(d * np.ceil(b / d)) for b, d in zip(block_img_size, net_axes_in_div_by)]

    # check overlap valid
    overlap = [i if i > 1 else i * s for i, s in zip(overlap, block_img_size)]
    overlap = [0 if b >= s else i for i, b, s in zip(overlap, block_img_size, x.shape)]
    overlap = [i if i % 2 == 0 else i + 1 for i in overlap]

    block_img_size = [int(b - 2 * i) for b, i in zip(block_img_size, overlap)]

    im_size_padded = [((g + 1) * b if o != 0 else g * b) for g, b, o in zip(block_size, block_img_size, overlap)]

    def pad_list_to_valid(input_list, target_shape):
        for idx, (value, target) in enumerate(zip(input_list, target_shape)):
            if value < target:
                input_list[idx] = (target + 7) // 8 * 8

        return input_list

    im_size_padded = pad_list_to_valid(im_size_padded, [i + o // 2 for o, i in zip(overlap, x.shape)])

    pad = {
        a: (o // 2, k - i - o // 2)
        for a, o, i, k in zip(axes, overlap, x.shape, im_size_padded)
    }

    crop = tuple(
        slice(p[0] * model.factor, -p[1] * model.factor if p[1] > 0 else None)
        for p in (pad[a] for a in axes)
    )

    x_pad = np.pad(x, tuple(pad[a] for a in axes), mode="reflect")
    # tifffile.imsave(fr"D:\VSproject\pad.tif", x_pad, imagej=True)

    logger.info(f"Pad img {x.shape} => {x_pad.shape} with pad: {pad}")
    sr_size = [i * model.factor for i in x_pad.shape]
    ret = np.zeros(sr_size, dtype=np.float32)
    if method == "mean":
        count = np.zeros(sr_size, dtype=np.int32)

    input_slice, output_slice, crop_slice = slice_generator(x_pad, block_img_size, overlap, model.factor)

    logger.info(f"predict start with {len(input_slice)} patches...")
    for pdx, (i_slice, o_slice, crop_slice) in enumerate(zip(input_slice, output_slice, crop_slice)):
        patch_start = time.time()

        logger.debug(
            f"get patch: i_slice: {slice2str(i_slice)}, o_slice: {slice2str(o_slice)}, crop_slice: {slice2str(crop_slice)}")
        block = x_pad[i_slice]
        if opt['dn_label'] and not opt['sr_label']:
            predict_fn = getattr(model, f"_{model.__class__.__name__}__predict_block_denoise")
        elif not opt['dn_label'] and opt['sr_label']:
            predict_fn = getattr(model, f"_{model.__class__.__name__}__predict_block_sr")
        else:
            predict_fn = getattr(model, f"_{model.__class__.__name__}__predict_block_all")

        out = predict_fn(block)

        # linear blending
        for i_a in range(x.ndim):
            if i_slice[i_a].start != 0:
                blending_axes = i_a
                logger.debug(f"blending axes: {blending_axes}")
                for i in range(overlap[blending_axes] * model.factor):
                    src_weight = (i + 1) / (overlap[blending_axes] * model.factor + 1)
                    dst_weight = 1 - src_weight
                    blending_slice = tuple(
                        [crop_slice[j] if j != blending_axes else crop_slice[j].start + i for j in range(x.ndim)])
                    ret_slice = tuple(
                        [o_slice[j] if j != blending_axes else o_slice[j].start + i for j in range(x.ndim)])
                    out[blending_slice] = src_weight * out[blending_slice] + dst_weight * ret[ret_slice]

        if method == "mean":
            count[o_slice] += 1
            ret[o_slice] += out[crop_slice]
        else:
            ret[o_slice] = out[crop_slice]

        patch_end = time.time()
        patch_time = patch_end - patch_start
        logger.info(f"patch [{pdx + 1:02d}]/ [{len(input_slice)}] shape: {block.shape} time: {patch_time:.4f}s")

    # torch.cuda.empty_cache()
    if method == "mean":
        return ret[crop] / count[crop]
    else:
        return ret[crop]


def _interpolate_z_for_sr(input_volume, z_upscale_factor, stage_tag):
    logger.info(">>> %s: Upsampling Z-axis by factor %.6f...", stage_tag, z_upscale_factor)
    sr_input = zoom(input_volume, (z_upscale_factor, 1, 1), order=1)
    logger.info("    Shape change: %s -> %s", input_volume.shape, sr_input.shape)
    return sr_input


def reverse_and_save_img(img, mi, ma, save_img_path, save_mip_path, scale=1, mode="16bit"):
    if img is None:
        return

    # time.sleep(5)  # Test non-blocking thread
    if img.ndim == 3:
        img = np.expand_dims(img, 1)

    alpha = ma - mi
    beta = mi

    if mode == "8bit":
        if ma > 255:
            alpha = 255 / ma * beta

        img = alpha * img + beta
        img = img.astype(np.uint8)
    elif mode == "16bit":
        img = np.clip(img, 0, 65535)
        # img = scale * alpha * img + beta
        img = alpha * img + beta
        img = img.astype(np.uint16)

    tifffile.imwrite(save_img_path, img, imagej=True)

    if save_mip_path:
        mip_xy = np.max(img, 0).squeeze()
        imageio.imsave(save_mip_path, mip_xy)

    return True


def main(opt):
    torch.cuda.empty_cache()
    model = create_model(opt)

    threads = []
    # 1. Save the original configuration
    original_dn_label = opt['dn_label']
    original_sr_label = opt['sr_label']
    original_scale = model.factor

    for phase, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
        test_set_name = test_loader.dataset.opt['name']
        logger.info('Testing [{:s}]...'.format(test_set_name))

        load_data_start = time.time()
        predict_time_start = time.time()

        for idx, (test_data, mi, ma) in enumerate(test_loader):
            tiff_time_start = time.time()
            tiff_file_path = test_set.paths_LQ[idx]
            test_img = test_data[0].numpy()

            # Initial dataset scaling (usually unchanged)
            # if dataset_opt['zoom_scale'] and dataset_opt['zoom_scale'] > 1:
            #     test_img = zoom(test_img, (dataset_opt['zoom_scale'], 1, 1), order=1)

            mi = mi[0].numpy()
            ma = ma[0].numpy()
            logger.info(f"Processing [{idx + 1}]/[{len(test_loader)}]: {tiff_file_path}...")

            load_data_end = time.time()
            load_data_time = load_data_end - load_data_start
            logger.info(f"Loading data time: {load_data_time:.4f}s")

            # === Prepare paths and filenames ===
            path_dict = model.opt.get('path', {})
            pretrain_path_SR = path_dict.get('pretrain_model_SR')
            pretrain_path_C = path_dict.get('pretrain_model_C')
            model_path_ref = pretrain_path_SR if pretrain_path_SR else (
                pretrain_path_C if pretrain_path_C else "Unknown/Unknown.pth")

            normalized_path = os.path.normpath(model_path_ref)
            path_parts = normalized_path.split(os.path.sep)
            model_filename = path_parts[-1]
            model_basename = os.path.splitext(model_filename)[0]
            if len(model_basename) > 2: model_basename = model_basename[:-2]

            # Create the output folder during the first iteration
            base_dir = test_loader.dataset.opt['dataroot_LQ']
            from datetime import datetime
            now = datetime.now()
            formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
            custom_time = formatted_time.replace(" ", "_").replace(":", "_").replace("-", "_")

            if idx == 0:
                save_img_base = os.path.join(base_dir, custom_time)
                os.makedirs(save_img_base, exist_ok=True)
                with open(os.path.join(save_img_base, 'label.txt'), "w") as file:
                    file.write(path_parts[-3] if len(path_parts) >= 3 else "UnknownModel")

            # === Decide whether split mode is needed ===
            is_split_mode = original_dn_label and original_sr_label
            final_result = None

            z_upscale_factor = dataset_opt['zoom_scale']

            if is_split_mode:
                # ==========================================
                # Stage 1: Run denoising (no upsampling)
                # ==========================================
                logger.info(">>> [Split Mode] Stage 1: Running Denoise...")

                opt['dn_label'] = True
                opt['sr_label'] = False
                model.factor = 1  # Denoising does not change the spatial size

                denoise_out = None
                success = False
                failed_count = 0
                net_axes_in_div_by = [opt['val'].get('min_devide_by', 8)] * test_img.ndim
                block_size = list(model.block_size)

                while not success:
                    try:
                        denoise_out = predict_tile(model, test_img, net_axes_in_div_by, block_size, model.over_lap)
                        torch.cuda.empty_cache()
                        success = True
                    except RuntimeError as e:
                        torch.cuda.empty_cache()
                        block_size = [block_size[i] if i != test_img.ndim - 1 - failed_count else block_size[i] + 1 for
                                      i in range(test_img.ndim)]
                        failed_count = (failed_count + 1) % test_img.ndim
                        logger.error(f"Stage 1 Error, retry: {block_size}...\n{e}")

                # ==========================================
                # Stage 1.5: Z-axis upsampling (intermediate processing)
                # ==========================================
                sr_input = _interpolate_z_for_sr(
                    denoise_out,
                    z_upscale_factor,
                    "[Split Mode] Intermediate",
                )

                # ==========================================
                # Stage 2: Run SR using the Z-upsampled image as input
                # ==========================================
                logger.info(">>> [Split Mode] Stage 2: Running SR...")

                opt['dn_label'] = False
                opt['sr_label'] = True
                model.factor = original_scale  # Restore the SR scale factor for XY output size calculation

                success = False
                failed_count = 0
                block_size = list(model.block_size)

                while not success:
                    try:
                        # Note: sr_input is already enlarged along Z, and predict_tile will tile it automatically
                        final_result = predict_tile(model, sr_input, net_axes_in_div_by, block_size, model.over_lap)
                        success = True
                    except RuntimeError as e:
                        torch.cuda.empty_cache()
                        block_size = [block_size[i] if i != test_img.ndim - 1 - failed_count else block_size[i] + 1 for
                                      i in range(test_img.ndim)]
                        failed_count = (failed_count + 1) % test_img.ndim
                        logger.error(f"Stage 2 Error, retry...\n{e}")

                task_suffix = '_SR_'
                fold_name = "SR"

            elif opt['sr_label']:
                logger.info(">>> [SR Mode] Preparing isotropic input for SR...")
                sr_input = _interpolate_z_for_sr(
                    test_img,
                    z_upscale_factor,
                    "[SR Mode] Intermediate",
                )
                net_axes_in_div_by = [opt['val'].get('min_devide_by', 8)] * test_img.ndim
                block_size, overlap = model.block_size, model.over_lap
                success = False
                failed_count = 0
                while not success:
                    try:
                        final_result = predict_tile(model, sr_input, net_axes_in_div_by, block_size, overlap)
                        success = True
                    except RuntimeError as e:
                        torch.cuda.empty_cache()
                        block_size = [block_size[i] if i != test_img.ndim - 1 - failed_count else block_size[i] + 1 for
                                      i in range(test_img.ndim)]
                        failed_count = (failed_count + 1) % test_img.ndim
                        logger.error(f"Runtime error, retry...\n{e}")
                task_suffix, fold_name = '_SR_', "SR"
            else:
                net_axes_in_div_by = [opt['val'].get('min_devide_by', 8)] * test_img.ndim
                block_size, overlap = model.block_size, model.over_lap
                success = False
                failed_count = 0
                while not success:
                    try:
                        final_result = predict_tile(model, test_img, net_axes_in_div_by, block_size, overlap)
                        success = True
                    except RuntimeError as e:
                        torch.cuda.empty_cache()
                        block_size = [block_size[i] if i != test_img.ndim - 1 - failed_count else block_size[i] + 1 for
                                      i in range(test_img.ndim)]
                        failed_count = (failed_count + 1) % test_img.ndim
                        logger.error(f"Runtime error, retry...\n{e}")
                task_suffix, fold_name = '_C_', "Denoise"

            # === Save the final result ===
            # Restore the original state
            opt['dn_label'] = original_dn_label
            opt['sr_label'] = original_sr_label
            model.factor = original_scale

            img_name = model_basename + task_suffix + os.path.split(tiff_file_path)[-1]
            save_img_dir = os.path.join(save_img_base, fold_name)
            os.makedirs(save_img_dir, exist_ok=True)
            t = threading.Thread(target=reverse_and_save_img,
                                 args=(final_result, mi, ma, os.path.join(save_img_dir, img_name),
                                       None, model.factor, model.mode))
            t.daemon = True
            t.start()
            threads.append(t)

            load_data_start = time.time()
            tiff_time_end = time.time()
            logger.info(f"Tiff completed! time: {tiff_time_end - tiff_time_start:.4f}s")

        predict_time_end = time.time()
        logger.info(f"Dataset [{test_set_name}] completed! time: {predict_time_end - predict_time_start:.4f}s")

    for t in threads:
        if t.is_alive():
            t.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HBExMNet inference launcher")
    parser.add_argument("-opt", type=str, default="options/test/test_Denoise_SR_auto.yml", help="Path to the YAML config file.")
    parser.add_argument("--mode", choices=["TIM", "SDCM"])
    parser.add_argument("--organelle")
    parser.add_argument("--task", choices=["denoise", "sr", "both"])
    parser.add_argument("--input-path")
    parser.add_argument("--xy-nm", type=float)
    parser.add_argument("--z-nm", type=float)
    parser.add_argument("--z-zoom", type=float)
    args = parser.parse_args()

    selection = _build_inference_selection(args)
    opt = options.parse(args.opt, mode="Denoise_SR_v", inference_selection=selection)
    opt = options.dict_to_nonedict(opt)
    logger.info(f"Read data/model config from yaml file: {args.opt}")
    logger.info("Yaml params: \n" + options.dict2str(opt))

    for key, path in opt["path"].items():
        if isinstance(path, str) and key not in {"experiments_root", "root", "repo_root"} and not list_any_in_str(["pretrain_model", "resume"], key):
            if not os.path.exists(path):
                mkdirs(path)
                logger.info(f"Create {key} path: {path}")

    main(opt)
