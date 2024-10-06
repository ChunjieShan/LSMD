from ssd.data.datasets import VOCDataset, COCODataset
from ssd.data.datasets.coco_vid import COCOVIDDataset
from .coco import coco_evaluation
from .voc import voc_evaluation
from .coco_vid import cocovid_evaluation


def evaluate(dataset, predictions, start_idxs, output_dir, **kwargs):
    """evaluate dataset using different methods based on dataset type.
    Args:
        dataset: Dataset object
        predictions(list[(boxes, labels, scores)]): Each item in the list represents the
            prediction results for one image. And the index should match the dataset index.
        output_dir: output folder, to save evaluation files or results.
    Returns:
        evaluation result
    """
    if isinstance(dataset, VOCDataset) or isinstance(dataset, COCODataset):
        args = dict(
            dataset=dataset, predictions=predictions, output_dir=output_dir, **kwargs,
        )
        if isinstance(dataset, VOCDataset):
            return voc_evaluation(**args)
        elif isinstance(dataset, COCODataset):
            return coco_evaluation(**args)

    else:
        args = dict(
            dataset=dataset, predictions=predictions, start_idxs=start_idxs, output_dir=output_dir, **kwargs,
        )
        if isinstance(dataset, COCOVIDDataset):
            return cocovid_evaluation(**args)
        else:
            raise NotImplementedError