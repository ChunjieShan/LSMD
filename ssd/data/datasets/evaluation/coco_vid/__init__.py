import json
import logging
import os
from datetime import datetime
from ssd.utils.hidden_print import HiddenPrint


def coco_slice(coco, start_idx):
    start_ann = 0
    end_ann = 0
    anns = coco.anns
    imgs = coco.imgs
    dataset = coco.dataset
    dataset_img = dataset['images']
    dataset_ann = dataset['annotations']

    anns_slice = {}
    imgs_slice = {}

    for k, v in anns.items():
        if v['image_id'] == start_idx:
            start_ann = k
            break

    for k, v in anns.items():
        if v['image_id'] == start_idx + 32:
            end_ann = k

    for k in list(anns.keys())[start_ann:end_ann]:
        anns_slice[k] = anns[k]

    for k in list(imgs.keys())[start_idx:(start_idx + 32)]:
        imgs_slice[k] = imgs[k]

    dataset_img = dataset_img[start_idx:(start_idx + 32)]
    dataset_ann = dataset_ann[start_ann:end_ann]

    coco.dataset['images'] = dataset_img
    coco.dataset['annotations'] = dataset_ann

    coco.anns = anns_slice
    coco.imgs = imgs_slice

    return coco


def cocovid_evaluation(dataset, predictions, start_idxs, output_dir, iteration=None):
    from pycocotools.cocoeval import COCOeval
    from pycocotools.coco import COCO
    logger = logging.getLogger("SSD.Inference")
    coco_results = []

    for k, v in predictions.items():
        anno_path, video_prediction = k, v

    # for anno_path, start_idx in zip(predictions, start_idxs):
        print(anno_path)
        # video_prediction = predictions[anno_path]
        start_idx = start_idxs[0].tolist()[0]

        with HiddenPrint():
            coco_gt = COCO(anno_path)

        coco_gt = coco_slice(coco_gt, start_idx)

        ids = list(coco_gt.imgs.keys())
        coco_categories = sorted(coco_gt.getCatIds())
        coco_id_to_contiguous_id = {
            coco_id: i + 1 for i, coco_id in enumerate(coco_categories)}
        class_mapper = {v: k for k, v in coco_id_to_contiguous_id.items()}

        for img_idx, prediction in enumerate(video_prediction):
            curr_idx = img_idx + start_idx
            prediction = prediction.resize((300, 300)).numpy()
            boxes, labels, scores = prediction['boxes'], prediction['labels'], prediction['scores']
            if labels.shape[0] == 0:
                continue

            boxes = boxes.tolist()
            labels = labels.tolist()
            scores = scores.tolist()

            coco_results.extend(
                [
                    {
                        "image_id": curr_idx,
                        "category_id": labels[k],
                        "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                        "score": scores[k]
                    }
                    for k, box in enumerate(boxes)
                ])

        iou_type = 'bbox'
        json_result_file = os.path.join(output_dir, iou_type + ".json")

        with open(json_result_file, "w") as f:
            json.dump(coco_results, f)

        logger.info("Writing results to {}".format(json_result_file))

        coco_dt = coco_gt.loadRes(json_result_file)
        coco_eval = COCOeval(coco_gt, coco_dt, iou_type)

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    result_strings = []
    keys = ["AP", "AP50", "AP75", "APs", "APm", "APl"]

    metrics = {}

    for i, key in enumerate(keys):
        metrics[key] = coco_eval.stats[i]
        logger.info('{:<10}: {}'.format(key, round(coco_eval.stats[i], 3)))
        result_strings.append('{:<10}: {}'.format(
            key, round(coco_eval.stats[i], 3)))

    return dict(metrics=metrics)
