from operator import is_
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from ssd.data import samplers
from ssd.data.datasets import build_dataset
from ssd.data.transforms import build_transforms, build_target_transform, build_prev_clip_transform
from ssd.structures.container import Container


class BatchCollator:
    def __init__(self, is_train=True):
        self.is_train = is_train

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        # images = default_collate(transposed_batch[0])
        # prev_images, curr_images = default_collate(tuple(transposed_batch[0][0]["prev"])), default_collate(
        #     tuple(transposed_batch[0][0]["curr"]))
        batches_data = transposed_batch[0]
        prev_tensors_list = []
        curr_tensors_list = []
        for batch in batches_data:
            prev_tensors_list.append(batch["prev"])
            curr_tensors_list.append(batch["curr"])

        prev_tensor = default_collate(prev_tensors_list)
        curr_tensor = default_collate(curr_tensors_list)

        data = Container(
            prev=prev_tensor,
            curr=curr_tensor
        )

        list_targets = transposed_batch[1]

        targets = Container(
            {key: default_collate([d[key] for d in list_targets]) for key in list_targets[0]}
        )

        return data, targets

        # if self.is_train:
        #     img_ids = default_collate(transposed_batch[2])
        #     list_targets = transposed_batch[1][0]
        #     # list_orig_targets = transposed_batch[2][0]
        #     targets = Container(
        #         {key: default_collate([d[key] for d in list_targets]) for key in list_targets[0]}
        #     )
        #     # orig_targets = Container(
        #     #     {key: default_collate([d[key] for d in list_orig_targets]) for key in list_orig_targets[0]}
        #     # )
        #     return images, targets, img_ids
        # else:
        #     img_ids = default_collate(transposed_batch[3])
        #     list_targets = transposed_batch[1][0]
        #     targets = list_targets
        #     anno_path = default_collate(transposed_batch[2])
        #     return images, targets, img_ids, anno_path


def make_data_loader(cfg, is_train=True, distributed=False, max_iter=None, start_iter=0):
    prev_transform = build_prev_clip_transform(cfg, is_train=is_train)
    train_transform = build_transforms(cfg, is_train=is_train)
    target_transform = build_target_transform(cfg) if is_train else None
    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST
    frame_length = int(cfg.DATASETS.FRAME_LENGTH)
    datasets = build_dataset(dataset_list, transform=train_transform, prev_transform=prev_transform,
                             target_transform=target_transform,
                             is_train=is_train, frame_length=frame_length, num_prev_clips=cfg.SOLVER.NUM_PREV_CLIPS)

    shuffle = is_train

    data_loaders = []

    for dataset in datasets:
        if distributed:
            sampler = samplers.DistributedSampler(dataset, shuffle=False)
        elif shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.sampler.SequentialSampler(dataset)

        batch_size = cfg.SOLVER.BATCH_SIZE
        eval_batch_size = cfg.SOLVER.EVAL_BATCH_SIZE

        if is_train:
            batch_sampler = torch.utils.data.sampler.BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=False)
        else:
            batch_sampler = torch.utils.data.sampler.BatchSampler(sampler=sampler, batch_size=eval_batch_size, drop_last=False)

        # batch_sampler = torch.utils.data.sampler.BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=False)
        if max_iter is not None:
            batch_sampler = samplers.IterationBasedBatchSampler(batch_sampler, num_iterations=max_iter,
                                                                start_iter=start_iter)

        data_loader = DataLoader(dataset, num_workers=cfg.DATA_LOADER.NUM_WORKERS, batch_sampler=batch_sampler,
                                 pin_memory=cfg.DATA_LOADER.PIN_MEMORY, collate_fn=BatchCollator(is_train))
        data_loaders.append(data_loader)

    if is_train:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return data_loaders[0]

    return data_loaders
