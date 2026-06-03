import logging

from torch.utils.data import DataLoader

from .lq_dataset import LQDataset

logger = logging.getLogger("data")


def create_dataset(dataset_opt):
    mode = dataset_opt["mode"]
    if mode == "LQ":
        dataset = LQDataset(dataset_opt)
    else:
        raise NotImplementedError(f"Dataset mode [{mode}] is not supported.")

    logger.info("Dataset [%s - %s] is created.", dataset.__class__.__name__, dataset_opt["name"])
    return dataset


def create_dataloader(dataset, dataset_opt):
    is_train = dataset_opt.get("is_train", False)
    return DataLoader(
        dataset,
        batch_size=1 if not is_train else dataset_opt.get("batch_size", 1),
        shuffle=bool(is_train),
        num_workers=0,
        pin_memory=False,
    )
