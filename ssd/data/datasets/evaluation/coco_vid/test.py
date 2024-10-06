from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


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


if __name__ == "__main__":
    gt_file = "/home/scj/Projects/Code/Data/PSAX-A/PSAX_OD/Dataset_ssd_v9/annotations/test/USM.1.2.840.113619.2.391.3780.1637833189.232.1.512_AURORA-003780.DCM.json"
    dt_file = "/home/scj/Projects/Code/0.Frameworks/SSD/outputs/vgg_ssd300_coco_vid/inference/coco_vid_test/bbox.json"
    coco_gt = COCO(annotation_file="/home/scj/Projects/Code/Data/PSAX-A/PSAX_OD/Dataset_ssd_v9/annotations/test/USM.1.2.840.113619.2.391.3780.1637833189.232.1.512_AURORA-003780.DCM.json")
    # print(coco_gt.dataset)
    coco_dt = coco_gt.loadRes(
        "/home/scj/Projects/Code/0.Frameworks/SSD/outputs/vgg_ssd300_coco_vid/inference/coco_vid_test/bbox.json")
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    #
    coco_sliced = coco_slice(coco_gt, 14)
    coco_dt_sliced = coco_sliced.loadRes(dt_file)

    coco_eval = COCOeval(coco_sliced, coco_dt_sliced, "bbox")

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
