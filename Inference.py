import os.path
import time
import argparse
from datetime import datetime

import torch
import imageio
import numpy as np
import threading
import tifffile
from scipy.ndimage import zoom

from data import create_dataloader, create_dataset
from models import create_model
from options import options
from utils.files import mkdirs
from utils.logger import get_logger
from utils.util import flexible_downsample, list_any_in_str, slice2str

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


def _select_predict_fn(model, run_denoise, run_sr):
    if run_denoise and not run_sr:
        return getattr(model, f"_{model.__class__.__name__}__predict_block_denoise")
    if run_sr and not run_denoise:
        return getattr(model, f"_{model.__class__.__name__}__predict_block_sr")
    return getattr(model, f"_{model.__class__.__name__}__predict_block_all")


def predict_tile(model, x, net_axes_in_div_by, block_size, overlap, run_denoise, run_sr, method="cover"):
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
        predict_fn = _select_predict_fn(model, run_denoise=run_denoise, run_sr=run_sr)
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


def _run_tiled_prediction_with_retry(model, image, min_divide_by, run_denoise, run_sr):
    net_axes_in_div_by = [min_divide_by] * image.ndim
    block_size = list(model.block_size)
    failed_axis = 0

    while True:
        try:
            return predict_tile(
                model,
                image,
                net_axes_in_div_by,
                block_size,
                model.over_lap,
                run_denoise=run_denoise,
                run_sr=run_sr,
            )
        except RuntimeError as exc:
            torch.cuda.empty_cache()
            retry_axis = image.ndim - 1 - failed_axis
            block_size = [
                block_size[index] if index != retry_axis else block_size[index] + 1
                for index in range(image.ndim)
            ]
            failed_axis = (failed_axis + 1) % image.ndim
            logger.error("Runtime error, retry with block size %s.\n%s", block_size, exc)


def _resolve_output_root(base_dir, model_path_ref):
    now = datetime.now()
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
    save_img_base = os.path.join(base_dir, timestamp)
    os.makedirs(save_img_base, exist_ok=True)

    normalized_path = os.path.normpath(model_path_ref)
    path_parts = normalized_path.split(os.path.sep)
    model_label = path_parts[-3] if len(path_parts) >= 3 else "UnknownModel"
    with open(os.path.join(save_img_base, "label.txt"), "w", encoding="utf-8") as file:
        file.write(model_label)
    return save_img_base, path_parts


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

    if scale > 3 and img.ndim == 4:
        # Keep the final saved SR output at 3x in Z/X/Y relative to the stage-2 input,
        # regardless of the backbone's internal SR scale.
        save_scale = 3.0 / float(scale)
        img = flexible_downsample(img, (save_scale, 1, save_scale, save_scale))

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
            if len(model_basename) > 2:
                model_basename = model_basename[:-2]

            # Create the output folder during the first iteration
            base_dir = test_loader.dataset.opt['dataroot_LQ']
            if idx == 0:
                save_img_base, _ = _resolve_output_root(base_dir, model_path_ref)

            # === Decide whether split mode is needed ===
            is_split_mode = original_dn_label and original_sr_label
            final_result = None

            if is_split_mode:
                # ==========================================
                # Stage 1: Run denoising (no upsampling)
                # ==========================================
                logger.info(">>> [Split Mode] Stage 1: Running Denoise...")

                opt['dn_label'] = True
                opt['sr_label'] = False
                model.factor = 1  # Denoising does not change the spatial size

                denoise_out = _run_tiled_prediction_with_retry(
                    model,
                    test_img,
                    opt["val"].get("min_devide_by", 8),
                    run_denoise=True,
                    run_sr=False,
                )
                torch.cuda.empty_cache()

                # ==========================================
                # Stage 1.5: Z-axis upsampling (intermediate processing)
                # ==========================================
                z_upscale_factor = dataset_opt['zoom_scale']
                logger.info(f">>> [Split Mode] Intermediate: Upsampling Z-axis by factor {z_upscale_factor}...")

                # zoom arguments: (Z scale, H scale, W scale) -> (scale, 1, 1)
                # order=1 uses linear interpolation (bilinear/trilinear); order=0 is nearest-neighbor; order=3 is cubic
                sr_input = zoom(denoise_out, (z_upscale_factor, 1, 1), order=1)

                logger.info(f"    Shape change: {denoise_out.shape} -> {sr_input.shape}")

                # ==========================================
                # Stage 2: Run SR using the Z-upsampled image as input
                # ==========================================
                logger.info(">>> [Split Mode] Stage 2: Running SR...")

                opt['dn_label'] = False
                opt['sr_label'] = True
                model.factor = original_scale  # Restore the SR scale factor for XY output size calculation

                final_result = _run_tiled_prediction_with_retry(
                    model,
                    sr_input,
                    opt["val"].get("min_devide_by", 8),
                    run_denoise=False,
                    run_sr=True,
                )

                task_suffix = '_SR_'
                fold_name = "SR"

            else:
                # === Original logic (non-split mode) ===
                final_result = _run_tiled_prediction_with_retry(
                    model,
                    test_img,
                    opt["val"].get("min_devide_by", 8),
                    run_denoise=bool(opt["dn_label"]),
                    run_sr=bool(opt["sr_label"]),
                )

                if opt['sr_label']:
                    task_suffix, fold_name = '_SR_', "SR"
                elif opt['dn_label']:
                    task_suffix, fold_name = '_C_', "Denoise"
                else:
                    task_suffix, fold_name = '_Unknown_', "Unknown"

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
