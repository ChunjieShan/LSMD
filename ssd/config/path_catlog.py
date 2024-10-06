import os


class DatasetCatalog:
    # DATA_DIR = 'E:/Ultrasound/A4C_obj_det_chl_20220627/A4C/data/Dataset_of_ssd_v7/
    # DATA_DIR = '/home/scj/Projects/Code/Data/PSAX-A/PSAX_OD/New/new_data/Dataset_ssd_v1/'
    # DATA_DIR = '/home/scj/Code/Data/1.Echocardiogram/2.Data/PSAXMV/2.Labelled-Data/Object-Detection/single_frame_dataset/Dataset_ssd_v2'
    # DATA_DIR = '/home/scj/Code/Data/1.Echocardiogram/2.Data/PSAXMV/2.Labelled-Data/Object-Detection/single_frame_dataset/Dataset_ssd_v3'
    # DATA_DIR = '/home/scj/Code/Data/1.Echocardiogram/2.Data/PSAXA/single_frame_dataset/Dataset_ssd_v2'
    # DATA_DIR = '/home/scj/Code/Data/2.Carotid-Artery/2.Labelled-Data/2.OD-Labels/More-Frames-Data/20230123_in_ex_common/'
    # DATA_DIR = '/home/scj/Code/Data/2.Carotid-Artery/2.Labelled-Data/2.OD-Labels/More-Frames-Data/20230201/20230201_cross'
    # DATA_DIR = '/mnt/j/Dataset/2.Carotid-Artery/2.Object-Detection/1.Training/20230825/'
    DATA_DIR = '/mnt/j/Dataset/2.Carotid-Artery/2.Object-Detection/1.Training/20240725/'
    # DATA_DIR = '/home/scj/Code/Data/1.Echocardiogram/2.Data/A5C/2.Labelled-Data/'
    DATASETS = {
        'voc_2007_train': {
            "data_dir": "VOC2007",
            "split": "train"
        },
        'voc_2007_val': {
            "data_dir": "VOC2007",
            "split": "val"
        },
        'voc_2007_trainval': {
            "data_dir": "VOC2007",
            "split": "trainval"
        },
        'voc_2007_test': {
            "data_dir": "VOC2007",
            "split": "test"
        },
        'voc_2012_train': {
            "data_dir": "VOC2012",
            "split": "train"
        },
        'voc_2012_val': {
            "data_dir": "VOC2012",
            "split": "val"
        },
        'voc_2012_trainval': {
            "data_dir": "VOC2012",
            "split": "trainval"
        },
        'voc_2012_test': {
            "data_dir": "VOC2012",
            "split": "test"
        },
        'coco_2014_valminusminival': {
            "data_dir": "val2014",
            "ann_file": "annotations/instances_valminusminival2014.json"
        },
        'coco_2014_minival': {
            "data_dir": "val2014",
            "ann_file": "annotations/instances_minival2014.json"
        },
        'coco_2014_train': {
            "data_dir": "train2014",
            "ann_file": "annotations/instances_train2014.json"
        },
        'coco_2014_val': {
            "data_dir": "val2014",
            "ann_file": "annotations/instances_val2014.json"
        },
        'coco_ours_train': {
            "data_dir": "train",
            "ann_file": "annotations/instances_train.json"
        },
        'coco_ours_val': {
            "data_dir": "val",
            "ann_file": "annotations/instances_val.json"
        },
        'coco_ours_test': {
            "data_dir": "test",
            "ann_file": "annotations/instances_test.json"
        },
        'coco_vid_train': {
            "data_dir": "train",
            "ann_file": "annotations/instances_train.json"
        },
        'coco_vid_val': {
            "data_dir": "val",
            "ann_file": "annotations/instances_val.json"
        },
        'coco_vid_test': {
            "data_dir": "test",
            "ann_file": "annotations/instances_test.json"
        },
        'dark_vid_train': {
            "data_dir": "images/train",
            "ann_file": "labels"
        },
        'dark_vid_val': {
            "data_dir": "images/valid",
            "ann_file": "labels"
        },
        'dark_vid_test': {
            "data_dir": "images/valid",
            "ann_file": "labels"
        },
    }

    @staticmethod
    def get(name):
        if "voc" in name:
            voc_root = DatasetCatalog.DATA_DIR
            if 'VOC_ROOT' in os.environ:
                voc_root = os.environ['VOC_ROOT']

            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(voc_root, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(factory="VOCDataset", args=args)
        elif "coco" in name:
            if "vid" in name:
                coco_root = DatasetCatalog.DATA_DIR
                if 'COCO_ROOT' in os.environ:
                    coco_root = os.environ['COCO_ROOT']
                attrs = DatasetCatalog.DATASETS[name]
                args = dict(
                    data_dir=os.path.join(coco_root, attrs["data_dir"]),
                    ann_file=os.path.join(coco_root, attrs["ann_file"]),
                    csv_file=os.path.join(coco_root, "../", "labels")
                )
                return dict(factory="COCOVIDDataset", args=args)

            else:
                coco_root = DatasetCatalog.DATA_DIR
                if 'COCO_ROOT' in os.environ:
                    coco_root = os.environ['COCO_ROOT']

                attrs = DatasetCatalog.DATASETS[name]
                args = dict(
                    data_dir=os.path.join(coco_root, attrs["data_dir"]),
                    ann_file=os.path.join(coco_root, attrs["ann_file"]),
                )
                return dict(factory="COCODataset", args=args)

        elif "dark" in name:
            data_root = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_root, attrs["data_dir"]),
                label_dir=os.path.join(data_root, attrs["ann_file"]),
            )

            return dict(factory="DarkVIDDataset", args=args)

        raise RuntimeError("Dataset not available: {}".format(name))
