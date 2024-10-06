from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
# _C.MODEL.META_ARCHITECTURE = 'SSDDetector'
_C.MODEL.META_ARCHITECTURE = 'SSDXMemDetector'
# _C.MODEL.META_ARCHITECTURE = 'SSDTMADetector'
_C.MODEL.DEVICE = "cuda:0"
# match default boxes to any ground truth with jaccard overlap higher than a threshold (0.5)
_C.MODEL.THRESHOLD = 0.5
# _C.MODEL.NUM_CLASSES = 21
_C.MODEL.NUM_CLASSES = 4      # 目标类别数目，注意背景为类别1
# Hard negative mining
_C.MODEL.NEG_POS_RATIO = 3
_C.MODEL.CENTER_VARIANCE = 0.1
_C.MODEL.SIZE_VARIANCE = 0.2

# ---------------------------------------------------------------------------- #
# Backbone
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()
# _C.MODEL.BACKBONE.NAME = 'slowfast'
_C.MODEL.BACKBONE.NAME = 'MobileNetV3'
_C.MODEL.BACKBONE.OUT_CHANNELS = (512, 1024, 512, 256, 256, 256)
# _C.MODEL.BACKBONE.PRETRAINED = "checkpoints/slowfast_r50_256p_4x16x1_256e_kinetics400_rgb_20200728-145f1097.pth"
_C.MODEL.BACKBONE.PRETRAINED = ""
_C.MODEL.BACKBONE.WIDTH = 1.0
_C.MODEL.BACKBONE.DEPTH = 1.0


# -----------------------------------------------------------------------------
# PRIORS
# -----------------------------------------------------------------------------
_C.MODEL.PRIORS = CN()
_C.MODEL.PRIORS.FEATURE_MAPS = [38, 19, 10, 5, 3, 1]
_C.MODEL.PRIORS.STRIDES = [8, 16, 32, 64, 100, 300]
_C.MODEL.PRIORS.MIN_SIZES = [30, 60, 111, 162, 213, 264]
_C.MODEL.PRIORS.MAX_SIZES = [60, 111, 162, 213, 264, 315]
_C.MODEL.PRIORS.ASPECT_RATIOS = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
# When has 1 aspect ratio, every location has 4 boxes, 2 ratio 6 boxes.
# #boxes = 2 + #ratio * 2
# number of boxes per feature map location
_C.MODEL.PRIORS.BOXES_PER_LOCATION = [4, 6, 6, 6, 4, 4]
_C.MODEL.PRIORS.CLIP = True

# -----------------------------------------------------------------------------
# Box Head
# -----------------------------------------------------------------------------
_C.MODEL.BOX_HEAD = CN()
_C.MODEL.BOX_HEAD.NAME = 'SSDBoxHead'
# _C.MODEL.BOX_HEAD.NAME = 'SSDRPNHead'
_C.MODEL.BOX_HEAD.PREDICTOR = 'SSDBoxPredictor'

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Image size
# _C.INPUT.IMAGE_SIZE = 300
_C.INPUT.IMAGE_SIZE = 300    # 输入图像尺寸
# Values to be used for image normalization, RGB layout
_C.INPUT.PIXEL_MEAN = [123, 117, 104]

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# Sampling rate of a single batch
_C.DATASETS.FRAME_LENGTH = 3
# Previous video clip memory
_C.DATASETS.SEQUENCE_NUM = 2
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ()
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ()

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATA_LOADER = CN()
# Number of data loading threads
_C.DATA_LOADER.NUM_WORKERS = 4
_C.DATA_LOADER.PIN_MEMORY = False

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# train configs
_C.SOLVER.NUM_PREV_CLIPS = 2
_C.SOLVER.MAX_ITER = 120000
_C.SOLVER.LR_STEPS = [80000, 100000]
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.BATCH_SIZE = 4
_C.SOLVER.EVAL_BATCH_SIZE = 1
_C.SOLVER.LR = 1e-3
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 5e-4
_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITERS = 10

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.NMS_THRESHOLD = 0.5
_C.TEST.CONFIDENCE_THRESHOLD = 0.01
_C.TEST.MAX_PER_CLASS = -1
_C.TEST.MAX_PER_IMAGE = 30
_C.TEST.BATCH_SIZE = 4

_C.OUTPUT_DIR = 'outputs'
