import torch
import torch.utils.data
import logging
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import tqdm

from ssd.config import cfg
from pathlib import Path
from ssd.utils import dist_util
from ssd.utils.dist_util import synchronize
from ssd.utils.logger import setup_logger
from inference.ssd_xmem_infer_detector import SSDXMemInference
from ssd.data.datasets.dark_xmem_lst_vid import DarkVIDDataset
from ssd.config.path_catlog import DatasetCatalog
from ssd.data.build import make_data_loader
from ssd.modeling.detector import build_detection_model
from ssd.utils.checkpoint import CheckPointer

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')


def smooth(y, f=0.05):
    # Box filter of fraction f
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')  # y-smoothed


def do_evaluation(cfg, model, distributed=False, iteration=0):
    video_root = os.path.join(DatasetCatalog.DATA_DIR, 'videos/valid')
    class_names = DarkVIDDataset.classes
    class_maps = {k: v for k, v in enumerate(class_names)}
    class_maps[0] = "background"

    logger = logging.getLogger("SSD.Evaluator")
    logger.info("Start Evaluating...")
    TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'  # tqdm bar format
    # template = ('%22s' + '%11s' * 6) % ('Progress', 'Mem', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
    template = ('%22s' + '%11s' * 2) % ('Progress', 'Mem', 'Instances')
    eval_results = []
    eval_targets = []
    iouv = torch.linspace(0.5, 0.95, 10, device="cpu")  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    valid_videos_list = os.listdir(video_root)
    data_loaders_val = make_data_loader(cfg, is_train=False, distributed=distributed)[0]
    device = torch.device(cfg.MODEL.DEVICE)

    # pbar = tqdm.tqdm(valid_videos_list, desc=template, bar_format=TQDM_BAR_FORMAT)
    pbar = tqdm.tqdm(data_loaders_val, desc=template, bar_format=TQDM_BAR_FORMAT)

    # for i, video_name in enumerate(pbar):
    for i, (data, targets) in enumerate(pbar):
        # video_path = os.path.join(video_root, video_name)
        # results, targets = model.evaluate(video_path)
        curr_images, prev_images = data["curr"], data["prev"]
        curr_images, prev_images = curr_images.to(device), prev_images.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            _, results = model(prev_images, curr_images, targets=targets)
        cpu_results = []
        for result in results:
            # result["boxes"] = result["boxes"].to("cpu")
            # result["labels"] = result["boxes"].to("cpu")
            # result["scores"] = result["boxes"].to("cpu")
            result = result.to("cpu")
            cpu_results.append(result)

        targets = targets.to("cpu")
        eval_targets.append(targets)

        mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
        # pbar.set_description(('%11s' * 2 + '%11.4g' * 1) %
        #                      (f'{i}/{len(valid_videos_list) - 1}', mem, len(targets["labels"])))
        pbar.set_description(('%11s' * 2 + '%11.4g' * 1) %
                             (f'{i}/{len(data_loaders_val) - 1}', mem, len(targets["labels"])))
        eval_results.append(cpu_results)

    stats = []
    for j, curr_result in enumerate(eval_results):
        targets = eval_targets[j]
        gt_boxes = targets["boxes"]
        gt_labels = targets["labels"]

        for i, result in enumerate(curr_result):
            boxes = result['boxes'].detach().cpu().numpy()
            labels = result['labels'].detach().cpu().numpy()
            confs = result['scores'].detach().cpu().numpy()

            if len(boxes) == 0:
                continue
            # mask = confs > 0.1
            # boxes = boxes[mask]
            # labels = labels[mask]
            # confs = confs[mask]

            # gt_boxes = targets["boxes"].detach().cpu().numpy()[i].tolist()
            # gt_labels = targets["labels"].detach().cpu().numpy()[i].tolist()

            curr_sample_results = []
            curr_sample_gts = []
            for label_idx, label in enumerate(labels):
                box = boxes[label_idx, :]
                conf = confs[label_idx]
                curr_result = np.array([*box, conf, label])
                curr_sample_results.append(curr_result)

            for label_idx, gt_label in enumerate(gt_labels[i]):
                gt_box = gt_boxes[i][label_idx]
                gt_box = [part * 320.0 for part in gt_box]
                # gt_box = [part for part in gt_box]
                curr_sample_gt = np.array([gt_label, *gt_box])
                curr_sample_gts.append(curr_sample_gt)

            # curr_sample_results = np.array(curr_sample_results)
            # curr_sample_gts = np.array(curr_sample_gts)
            if len(curr_sample_gts) == 0:
                continue
            curr_sample_results = torch.Tensor(curr_sample_results)
            curr_sample_gts = torch.Tensor(curr_sample_gts)

            correct = process_batch(curr_sample_results, curr_sample_gts, iouv)
            stats.append((correct, curr_sample_results[:, 4], curr_sample_results[:, 5], curr_sample_gts[:, 0]))

    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False, save_dir=cfg.OUTPUT_DIR,
                                                  names=class_maps)
    # tp, fp, p, r, f1, ap, ap_class = ap_per_class_wo_plot(*stats, plot=False, save_dir=cfg.OUTPUT_DIR,
    #                                                       names=class_maps)
    ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
    mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    logger.info(f"Iteration: {iteration}")
    logger.info(f"Precision: {mp}")
    logger.info(f"Recall: {mr}")
    logger.info(f"AP50: {map50}")
    logger.info(f"AP@0.5:0.95: {map}")
    return map


def plot_pr_curve(px, py, ap, save_dir=Path('pr_curve.png'), names=()):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title('Precision-Recall Curve')
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)


def ap_per_class_wo_plot(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=(), eps=1e-16, prefix=""):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    # precisions, recalls = np.zeros((nc, ), dtype=np.float32), np.zeros((nc, ), dtype=np.float32)
    precisions, recalls = [], []
    p_ap50, r_ap50 = [], []
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).astype(np.float32)
        tpc = (tp[i]).astype(np.float32)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve

        precisions.append(precision)
        recalls.append(recall)
        p_ap50.append(precision[:, 0])
        r_ap50.append(recall[:, 0])

    # Compute F1 (harmonic mean of precision and recall)
    # f1 = 2 * precision * r / (p + r + eps)
    # list: only classes that have data
    # names = [v for k, v in names.items() if k in unique_classes]
    # names = dict(enumerate(names))  # to dict
    #
    # i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    # p, r, f1 = p[:, i], r[:, i], f1[:, i]
    # tp = (r * nt).round()  # true positives
    # fp = (tp / (p + eps) - tp).round()  # false positives
    # return tp, fp, p, r, f1, ap, unique_classes.astype(int)
    return


def plot_mc_curve(px, py, save_dir=Path('mc_curve.png'), names=(), xlabel='Confidence', ylabel='Metric'):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)

    y = smooth(py.mean(0), 0.05)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title(f'{ylabel}-Confidence Curve')
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)


def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    np_iou = iou.detach().cpu().numpy()
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=(), eps=1e-16, prefix=""):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if plot and j == 0:
                py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = dict(enumerate(names))  # to dict
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / f'{prefix}PR_curve.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / f'{prefix}F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / f'{prefix}P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / f'{prefix}R_curve.png', names, ylabel='Recall')

    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype(int)


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


def evaluation(cfg, ckpt, distributed):
    logger = logging.getLogger("SSD.inference")

    model = build_detection_model(cfg).eval()
    checkpointer = CheckPointer(model, resume=False, save_dir=cfg.OUTPUT_DIR, logger=logger)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    # model = SSDXMemInference(cfg=cfg, weights=ckpt, device=device, score_th=0.2)
    checkpointer.load(ckpt, use_latest=ckpt is None)
    do_evaluation(cfg, model, distributed)


def main():
    parser = argparse.ArgumentParser(description='SSD Evaluation on VOC and COCO dataset.')
    parser.add_argument(
        "--config-file",
        default='./configs/resnet/dark/darknet_ssd320_mem_l.yaml',
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--conf_threshold", type=float, default=0.2)
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default='./outputs/240103_carotid_l/best_map_model_49.pth',
        type=str,
    )

    parser.add_argument("--output_dir", default="eval_results", type=str,
                        help="The directory to store evaluation results.")

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if torch.cuda.is_available():
        # This flag allows you to enable the inbuilt cudnn auto-tuner to
        # find the best algorithm to use for your hardware.
        torch.backends.cudnn.benchmark = True
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.TEST.CONFIDENCE_THRESHOLD = args.conf_threshold
    cfg.freeze()

    logger = setup_logger("SSD", dist_util.get_rank(), cfg.OUTPUT_DIR)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    evaluation(cfg, ckpt=args.ckpt, distributed=distributed)


if __name__ == '__main__':
    main()
