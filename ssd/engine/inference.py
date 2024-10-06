import logging
import os

import torch
import torch.utils.data
from tqdm import tqdm

from ssd.data.build import make_data_loader
from ssd.data.datasets.evaluation import evaluate

from ssd.utils import dist_util, mkdir
from ssd.utils.dist_util import synchronize, is_main_process


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = dist_util.all_gather(predictions_per_gpu)
    if not dist_util.is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    # if len(image_ids) != image_ids[-1] + 1:
    #     logger = logging.getLogger("SSD.inference")
    #     logger.warning(
    #         "Number of images that were gathered from multiple processes is not "
    #         "a contiguous set. Some images might be missing from the evaluation"
    #     )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def compute_on_dataset(model, data_loader, device):
    results_dict = {}
    start_idxs = []
    anno_path = []
    for batch in tqdm(data_loader):
        videos, targets, start_idx, anno_path = batch
        # print(targets)
        cpu_device = torch.device("cpu")
        with torch.no_grad():
            videos = videos.permute(0, 2, 1, 3, 4)
            outputs = model(videos.to(device))[0]

            outputs = [o.to(cpu_device) for o in outputs]

            results_dict[anno_path[0]] = outputs
            start_idxs.append(start_idx)
            # results_dict.update(
            #     {anno_path[0]: result for video_id, result in zip(video_ids, outputs)}
            # )
    return results_dict, start_idxs


def inference(model, data_loader, dataset_name, device, output_folder=None, use_cached=False, **kwargs):
    dataset = data_loader.dataset
    logger = logging.getLogger("SSD.inference")
    logger.info("Evaluating {} dataset({} videos):".format(dataset_name, len(dataset)))
    predictions_path = os.path.join(output_folder, 'predictions.pth')
    if use_cached and os.path.exists(predictions_path):
        predictions = torch.load(predictions_path, map_location='cpu')
    else:
        predictions, start_idxs = compute_on_dataset(model, data_loader, device)
        synchronize()
        if "vid" in dataset_name:
            pass 
        else:
            predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return
    if output_folder:
        torch.save(predictions, predictions_path)
    return evaluate(dataset=dataset, predictions=predictions, start_idxs=start_idxs, output_dir=output_folder, **kwargs)


@torch.no_grad()
def do_evaluation(cfg, model, distributed, **kwargs):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    model.eval()
    device = torch.device(cfg.MODEL.DEVICE)
    data_loaders_val = make_data_loader(cfg, is_train=False, distributed=distributed)
    eval_results = []

    for dataset_name, data_loader in zip(cfg.DATASETS.TEST, data_loaders_val):
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        if not os.path.exists(output_folder):
            mkdir(output_folder)
        eval_result = inference(model, data_loader, dataset_name, device, output_folder, **kwargs)
        eval_results.append(eval_result)
        
    return eval_results
