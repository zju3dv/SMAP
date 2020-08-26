import math

import torch
import torchvision.transforms as transforms

from cvpack.dataset import torch_samplers

from dataset.data_settings import load_dataset
from dataset.base_dataset import JointDataset


def get_train_loader(cfg, num_gpu, is_dist=True, is_shuffle=True, start_iter=0, 
                     use_augmentation=True, with_mds=False):
    # -------- get raw dataset interface -------- #
    normalize = transforms.Normalize(mean=cfg.INPUT.MEANS, std=cfg.INPUT.STDS)
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    if cfg.DATASET.NAME == 'MIX':
        Dataset = JointDataset
    else:
        raise NameError("Dataset is not defined!", cfg.DATASET.NAME)

    dataset = Dataset(cfg, 'train', transform, use_augmentation, with_mds)

    # -------- make samplers -------- #
    if is_dist:
        sampler = torch_samplers.DistributedSampler(
                dataset, shuffle=is_shuffle)
    elif is_shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)

    images_per_gpu = cfg.SOLVER.IMG_PER_GPU

    aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []
    if aspect_grouping:
        batch_sampler = torch_samplers.GroupedBatchSampler(
                sampler, dataset, aspect_grouping, images_per_gpu,
                drop_uneven=False)
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
                sampler, images_per_gpu, drop_last=False)

    batch_sampler = torch_samplers.IterationBasedBatchSampler(
            batch_sampler, cfg.SOLVER.MAX_ITER, start_iter)

    # -------- make data_loader -------- #
    class BatchCollator(object):
        def __init__(self, size_divisible):
            self.size_divisible = size_divisible

        def __call__(self, batch):
            transposed_batch = list(zip(*batch))
            images = torch.stack(transposed_batch[0], dim=0)
            valids = torch.stack(transposed_batch[1], dim=0)
            labels = torch.stack(transposed_batch[2], dim=0)
            rdepth = torch.stack(transposed_batch[3], dim=0)
            return images, valids, labels, rdepth

    data_loader = torch.utils.data.DataLoader(
            dataset, num_workers=cfg.DATALOADER.NUM_WORKERS,
            batch_sampler=batch_sampler,
            collate_fn=BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY), )

    return data_loader


def get_test_loader(cfg, num_gpu, local_rank, stage, use_augmentation=False, with_mds=False):
    # -------- get raw dataset interface -------- #
    normalize = transforms.Normalize(mean=cfg.INPUT.MEANS, std=cfg.INPUT.STDS)
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    if cfg.DATASET.NAME == 'MIX':
        Dataset = JointDataset
    else:
        raise NameError("Dataset is not defined!", cfg.DATASET.NAME)

    dataset = Dataset(cfg, stage, transform, use_augmentation, with_mds)

    # -------- split dataset to gpus -------- #
    num_data = dataset.__len__()
    num_data_per_gpu = math.ceil(num_data / num_gpu)
    st = local_rank * num_data_per_gpu
    ed = min(num_data, st + num_data_per_gpu)
    indices = range(st, ed)
    subset = torch.utils.data.Subset(dataset, indices)

    # -------- make samplers -------- #
    sampler = torch.utils.data.sampler.SequentialSampler(subset)

    images_per_gpu = cfg.TEST.IMG_PER_GPU

    batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_gpu, drop_last=False)

    # -------- make data_loader -------- #
    class BatchCollator(object):
        def __init__(self, size_divisible):
            self.size_divisible = size_divisible

        def __call__(self, batch):
            transposed_batch = list(zip(*batch))
            images = torch.stack(transposed_batch[0], dim=0)
            meta_data = torch.stack(transposed_batch[1], dim=0)
            img_path = transposed_batch[2]
            scale = transposed_batch[3]
            return images, meta_data, img_path, scale

    data_loader = torch.utils.data.DataLoader(
            subset, num_workers=cfg.DATALOADER.NUM_WORKERS,
            batch_sampler=batch_sampler,
            collate_fn=BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY), )

    return data_loader
