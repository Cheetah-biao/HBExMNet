from __future__ import annotations

import argparse
import logging
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import options.options as option
from data.util import augment
from models import create_model
from options.train.Config_train import chunking_data
from utils import util
from utils.project_paths import workspace_path


def init_dist(backend="nccl", **kwargs):
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn")
    rank = int(os.environ["RANK"])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def _parse_cli_args():
    parser = argparse.ArgumentParser(description="HBExMNet training launcher")
    parser.add_argument("--mode", choices=["TIM", "SDCM"])
    parser.add_argument("--organelle")
    parser.add_argument("--task", choices=["Denoise", "SR"])
    parser.add_argument("--hr-path")
    parser.add_argument("--lr-path")
    parser.add_argument("--patch-d", type=int)
    parser.add_argument("--patch-h", type=int)
    parser.add_argument("--patch-w", type=int)
    parser.add_argument("--factor", type=int)
    parser.add_argument("--launcher", choices=["none", "pytorch"], default="none")
    parser.add_argument("--max-iters", type=int, help="Override train.niter for quick validation.")
    parser.add_argument("--val-freq", type=int, help="Override validation frequency.")
    parser.add_argument("--save-freq", type=int, help="Override checkpoint frequency.")
    return parser.parse_args()


def _selection_from_args(args):
    keys = {
        "mode": args.mode,
        "organelle": args.organelle,
        "task": args.task,
        "hr_path": args.hr_path,
        "lr_path": args.lr_path,
        "patch_d": args.patch_d,
        "patch_h": args.patch_h,
        "patch_w": args.patch_w,
        "factor": args.factor,
    }
    selection = {key: value for key, value in keys.items() if value is not None}
    return selection or None


def _build_training_request(args):
    selection = _selection_from_args(args)
    return chunking_data(selection=selection)


def _load_training_arrays(training_data_file):
    training_data_file = Path(training_data_file)
    if not training_data_file.exists():
        raise FileNotFoundError(f"Training data file was not found: {training_data_file}")
    data = np.load(training_data_file, allow_pickle=True)
    return data["X"], data["Y"], data["Z"]


def _prepare_splits(x, y, z, include_mid):
    sample_count = len(x)
    if sample_count == 0:
        raise ValueError("No training samples were found in training_data.npz.")

    if sample_count == 1:
        train_indices = np.array([0])
        val_indices = np.array([], dtype=int)
    else:
        val_count = max(1, int(round(sample_count * 0.2)))
        val_count = min(val_count, sample_count - 1)
        train_indices = np.arange(0, sample_count - val_count)
        val_indices = np.arange(sample_count - val_count, sample_count)

    train_set = [x[train_indices], y[train_indices]]
    val_set = [x[val_indices], y[val_indices]]
    if include_mid:
        train_set.append(z[train_indices])
        val_set.append(z[val_indices])
    return train_set, val_set


def _safe_crop_pair(pred, target, border):
    if border <= 0:
        return pred, target

    spatial_shape = pred.shape[-3:]
    if any(size <= border * 2 for size in spatial_shape):
        return pred, target

    crop = (slice(None), slice(None), slice(border, -border), slice(border, -border), slice(border, -border))
    return pred[crop], target[crop]


def _tensor_to_image(visuals, key):
    return util.tensor2img(visuals[key])


def _validation_keys(is_sr):
    if is_sr:
        return "SR_img", "LSNR"
    return "SR", "LR"


def _sample_train_batch(train_set, batch_size, include_mid):
    train_count = len(train_set[0])
    replace = train_count < batch_size
    indices = np.random.choice(train_count, size=batch_size, replace=replace)

    if include_mid:
        img_lq, img_mid, img_gt = train_set[1][indices], train_set[2][indices], train_set[0][indices]
        img_lq, img_mid, img_gt = augment([img_lq, img_mid, img_gt], True, True)
        return {
            "LQ": torch.from_numpy(np.ascontiguousarray(img_lq)).float(),
            "HQ": torch.from_numpy(np.ascontiguousarray(img_mid)).float(),
            "GT": torch.from_numpy(np.ascontiguousarray(img_gt)).float(),
        }

    img_lq, img_gt = train_set[1][indices], train_set[0][indices]
    img_lq, img_gt = augment([img_lq, img_gt], True, True)
    return {
        "LQ": torch.from_numpy(np.ascontiguousarray(img_lq)).float(),
        "GT": torch.from_numpy(np.ascontiguousarray(img_gt)).float(),
    }


def _sample_val_batch(val_set, index, include_mid):
    if include_mid:
        return {
            "LQ": torch.from_numpy(np.ascontiguousarray(np.expand_dims(val_set[1][index], axis=0))).float(),
            "HQ": torch.from_numpy(np.ascontiguousarray(np.expand_dims(val_set[2][index], axis=0))).float(),
            "GT": torch.from_numpy(np.ascontiguousarray(np.expand_dims(val_set[0][index], axis=0))).float(),
        }
    return {
        "LQ": torch.from_numpy(np.ascontiguousarray(np.expand_dims(val_set[1][index], axis=0))).float(),
        "GT": torch.from_numpy(np.ascontiguousarray(np.expand_dims(val_set[0][index], axis=0))).float(),
    }


def _setup_training(opt_path, model_name, launcher, overrides=None):
    opt = option.parse(opt_path, mode="Train", model_name=model_name)
    if overrides:
        if overrides.get("max_iters") is not None:
            opt["train"]["niter"] = int(overrides["max_iters"])
        if overrides.get("val_freq") is not None:
            opt["train"]["val_freq"] = int(overrides["val_freq"])
        if overrides.get("save_freq") is not None:
            opt["logger"]["save_checkpoint_freq"] = int(overrides["save_freq"])

    cpu_threads_num = opt["cpu_threads_num"]
    torch.set_num_threads(cpu_threads_num)

    if launcher == "none":
        opt["dist"] = False
        rank = -1
        print("Disabled distributed training.")
    else:
        opt["dist"] = True
        init_dist()
        rank = torch.distributed.get_rank()

    resume_state = None
    if opt["path"].get("resume_state", None):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt["path"]["resume_state"],
            map_location=lambda storage, loc: storage.cuda(device_id),
        )
        option.check_resume(opt, resume_state["iter"])

    if rank <= 0:
        if resume_state is None:
            util.mkdir_and_rename(opt["path"]["experiments_root"])
            util.mkdirs(
                (
                    path
                    for key, path in opt["path"].items()
                    if isinstance(path, str)
                    and key not in {"experiments_root", "root", "repo_root"}
                    and "pretrain_model" not in key
                    and "resume" not in key
                )
            )
        util.setup_logger("base", opt["path"]["log"], f"train_{model_name}", level=logging.INFO, screen=True, tofile=True)
        util.setup_logger("val", opt["path"]["log"], f"val_{model_name}", level=logging.INFO, screen=True, tofile=True)
        util.setup_logger(
            "train_psnr",
            opt["path"]["log"],
            f"train_psnr_{model_name}",
            level=logging.INFO,
            screen=True,
            tofile=True,
        )
        logger = logging.getLogger("base")
        logger.info(option.dict2str(opt))
    else:
        util.setup_logger("base", opt["path"]["log"], "train", level=logging.INFO, screen=True)
        logger = logging.getLogger("base")

    tb_logger = None
    if rank <= 0 and opt["use_tb_logger"] and "debug" not in model_name:
        version = float(torch.__version__[0:3])
        if version >= 1.1:
            from torch.utils.tensorboard import SummaryWriter
        else:
            from tensorboardX import SummaryWriter
        tb_logger = SummaryWriter(log_dir=str(workspace_path("tb_logger", model_name)))

    opt = option.dict_to_nonedict(opt)

    seed = opt["train"]["manual_seed"]
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info("Random seed: %s", seed)
    util.set_random_seed(seed)
    torch.backends.cudnn.benchmark = True

    return opt, logger, tb_logger, rank, resume_state


def _run_validation(model, val_set, opt, logger, current_step, epoch, tb_logger, model_name, rank, is_sr):
    if len(val_set[0]) == 0 or rank > 0:
        return None

    avg_psnr = 0.0
    sr_key, lr_key = _validation_keys(is_sr)
    save_test_data_num = 4

    for index in range(len(val_set[0])):
        val_data = _sample_val_batch(val_set, index, include_mid=is_sr)
        model.feed_data(val_data)
        model.test()
        visuals = model.get_current_visuals()

        gt_img = _tensor_to_image(visuals, "GT")
        sr_img = _tensor_to_image(visuals, sr_key)
        lr_img = _tensor_to_image(visuals, lr_key)

        if index < save_test_data_num:
            img_name = str(index)
            img_dir = os.path.join(opt["path"]["val_images"], img_name)
            util.mkdir(img_dir)
            util.save_img_t(sr_img, os.path.join(img_dir, f"{img_name}_SR_{current_step}.tif"))
            if current_step == opt["train"]["val_freq"]:
                util.save_img_t(gt_img, os.path.join(img_dir, f"{img_name}_GT_{current_step}.tif"))
                util.save_img_t(lr_img, os.path.join(img_dir, f"{img_name}_LR_{current_step}.tif"))

        bit = 65535.0
        crop_size = int(opt["scale"])
        gt_np = gt_img / bit
        sr_np = sr_img / bit
        if gt_np.ndim == 2:
            gt_np = np.expand_dims(gt_np, axis=0)
            sr_np = np.expand_dims(sr_np, axis=0)

        if crop_size > 0 and all(size > crop_size * 2 for size in gt_np.shape[-3:]):
            cropped_gt = gt_np[crop_size:-crop_size, crop_size:-crop_size, crop_size:-crop_size]
            cropped_sr = sr_np[crop_size:-crop_size, crop_size:-crop_size, crop_size:-crop_size]
        else:
            cropped_gt = gt_np
            cropped_sr = sr_np
        avg_psnr += util.calc_psnr_3d(cropped_sr * int(bit), cropped_gt * int(bit))

    avg_psnr /= len(val_set[0])
    logger.info("# Validation # PSNR: %.4e.", avg_psnr)
    logger_val = logging.getLogger("val")
    logger_val.info("<epoch:%3d, iter:%8d> psnr: %.4e.", epoch, current_step, avg_psnr)
    if tb_logger is not None and "debug" not in model_name:
        tb_logger.add_scalar("psnr", avg_psnr, current_step)
    return avg_psnr


def _run_training(opt_path, model_name, launcher, include_mid, training_data_file, overrides=None):
    opt, logger, tb_logger, rank, resume_state = _setup_training(opt_path, model_name, launcher, overrides=overrides)
    x, y, z = _load_training_arrays(training_data_file)
    opt["net_dim"] = len(x.shape)

    train_set, val_set = _prepare_splits(x, y, z, include_mid=include_mid)
    train_count = len(train_set[0])
    batch_size = int(opt["train"]["batch_size"])
    train_size = max(1, int(math.ceil(train_count / batch_size)))
    total_iters = int(opt["train"]["niter"])
    total_epochs = max(1, int(math.ceil(total_iters / train_size)))

    if rank <= 0:
        logger.info("Training data file: %s", training_data_file)
        logger.info("Number of train patches: %d, iters per epoch: %d", train_count, train_size)
        logger.info("Number of validation patches: %d", len(val_set[0]))
        logger.info("Total epochs needed: %d for %d iterations", total_epochs, total_iters)

    model = create_model(opt)
    if resume_state:
        logger.info("Resuming training from epoch: %s, iter: %s.", resume_state["epoch"], resume_state["iter"])
        start_epoch = resume_state["epoch"]
        current_step = resume_state["iter"]
        model.resume_training(resume_state)
    else:
        start_epoch = 0
        current_step = 0

    logger.info("Start training from epoch: %d, iter: %d", start_epoch, current_step)

    best_psnr = float("-inf")
    for epoch in range(start_epoch, total_epochs):
        epoch_psnr = 0.0
        epoch_steps = 0

        for _ in range(train_size):
            if current_step >= total_iters:
                break

            current_step += 1
            train_data = _sample_train_batch(train_set, batch_size, include_mid=include_mid)
            model.feed_data(train_data)
            model.optimize_parameters(current_step)
            model.update_learning_rate(current_step, warmup_iter=opt["train"]["warmup_iter"])

            if current_step % opt["logger"]["print_freq"] == 0 and rank <= 0:
                logs = model.get_current_log()
                message = f"<epoch:{epoch:3d}, iter:{current_step:8,d}, lr:{model.get_current_learning_rate():.3e}> "
                for key, value in logs.items():
                    message += f"{key}: {value:.4e} "
                    if tb_logger is not None and "debug" not in model_name:
                        tb_logger.add_scalar(key, value, current_step)
                logger.info(message)

            gt_tensor = train_data["GT"]
            pred_tensor = model.get_train_SR()
            if not isinstance(pred_tensor, torch.Tensor):
                pred_tensor = torch.as_tensor(pred_tensor)
            pred_tensor = pred_tensor.detach().cpu()
            gt_tensor = gt_tensor.detach().cpu()
            cropped_pred, cropped_gt = _safe_crop_pair(pred_tensor, gt_tensor, int(opt["scale"]))
            epoch_psnr += float(util.calc_psnr_3d_torch(cropped_pred * 65535, cropped_gt * 65535))
            epoch_steps += 1

            if current_step % opt["train"]["val_freq"] == 0:
                avg_psnr = _run_validation(model, val_set, opt, logger, current_step, epoch, tb_logger, model_name, rank, include_mid)
                if avg_psnr is not None and avg_psnr >= best_psnr:
                    best_psnr = avg_psnr
                    model.save("best")

            if current_step % opt["logger"]["save_checkpoint_freq"] == 0 and rank <= 0:
                logger.info("Saving models and training states.")
                model.save(current_step)
                model.save_training_state(epoch, current_step)

        if epoch_steps > 0 and rank <= 0:
            avg_psnr_train = epoch_psnr / epoch_steps
            logger.info("# Train # PSNR: %.4e.", avg_psnr_train)
            logger_train_psnr = logging.getLogger("train_psnr")
            logger_train_psnr.info("<epoch:%3d, iter:%8d> train_psnr: %.4e.", epoch, current_step, avg_psnr_train)
            if tb_logger is not None and "debug" not in model_name:
                tb_logger.add_scalar("train_psnr", avg_psnr_train, current_step)

        if current_step >= total_iters:
            break

    if rank <= 0:
        logger.info("Saving the final model.")
        model.save("latest")
        logger.info("End of %s training.", "SR" if include_mid else "Denoise")

    if tb_logger is not None:
        tb_logger.close()


def run_denoise_main(opt_path, model_name, training_data_file, launcher="none", overrides=None):
    _run_training(opt_path, model_name, launcher=launcher, include_mid=False, training_data_file=training_data_file, overrides=overrides)


def run_sr_main(opt_path, model_name, training_data_file, launcher="none", overrides=None):
    _run_training(opt_path, model_name, launcher=launcher, include_mid=True, training_data_file=training_data_file, overrides=overrides)


if __name__ == "__main__":
    args = _parse_cli_args()
    label_tag, factor, hr_path, lr_path, training_data_file = _build_training_request(args)

    print("--- Configuration Completed ---")
    print(f"Model Label: {label_tag}")
    print(f"Factor: {factor}")
    print(f"GT Path: {hr_path}")
    print(f"Raw Data Path: {lr_path}")
    print(f"Training Data File: {training_data_file}")

    overrides = {
        "max_iters": args.max_iters,
        "val_freq": args.val_freq,
        "save_freq": args.save_freq,
    }

    if label_tag.endswith("_Denoise"):
        run_denoise_main("options/train/train_denoise_auto.yml", label_tag, training_data_file, launcher=args.launcher, overrides=overrides)
    elif label_tag.endswith("_SR"):
        run_sr_main("options/train/train_SR_auto.yml", label_tag, training_data_file, launcher=args.launcher, overrides=overrides)
    else:
        raise ValueError(f"Could not infer the task from model label: {label_tag}")
