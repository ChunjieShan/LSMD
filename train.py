import argparse
import logging
import os

import torch
import torch.distributed as dist

from ssd.engine.inference import do_evaluation
from ssd.config import cfg
from ssd.data.build import make_data_loader
from ssd.engine.trainer import do_train
from ssd.modeling.detector import build_detection_model
from ssd.solver.build import make_optimizer, make_lr_scheduler
from ssd.utils import dist_util, mkdir
from ssd.utils.checkpoint import CheckPointer
from ssd.utils.dist_util import synchronize
from ssd.utils.logger import setup_logger
from ssd.utils.misc import str2bool


def train(cfg, args):
    logger = logging.getLogger('SSD.trainer')
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)

    # Loading pretrained weights
    if len(cfg.MODEL.BACKBONE.PRETRAINED) and not args.resume:
        logger.info("Loading pretrained weights...")
        model_dict = model.state_dict()
        pretrained_ckpt = torch.load(cfg.MODEL.BACKBONE.PRETRAINED)['model']

        new_state_dict = {}
        for k, v in pretrained_ckpt.items():
            if args.init_head:
                if "box_head" in k:
                    continue

            if "neck" in k:
                new_state_dict[k.replace(".neck", ".pan")] = v
            else:
                new_state_dict[k] = v
            # elif "encoder" in k:
            #     new_state_dict[k.replace(".encoder", ".attn.encoder").replace(".blocks.0", "")] = v
            #     # new_state_dict[k.replace("backbone.encoder", "ltam.encoder").replace(".blocks.0", "")] = v
            # elif "bottleneck_fc" in k:
            #     new_state_dict[k.replace(".decoder", ".attn.decoder").replace(".blocks.0", "")] = v
            # elif "decoder" in k:
            #     new_state_dict[k.replace(".decoder", ".attn.decoder").replace(".blocks.0", "")] = v
                # new_state_dict[k.replace("backbone.decoder", "ltam.decoder").replace(".blocks.0", "")] = v
            # if "bottleneck_fc" in k:
            #     new_state_dict[k.replace(".decoder", ".attn.decoder").replace(".blocks.0", "")] = v
                # new_state_dict[k.replace("backbone.decoder", "ltam.decoder").replace(".blocks.0", "")] = v
        missing_keys, unexpected_keys = model.load_state_dict(
            new_state_dict, strict=False)

        if len(unexpected_keys) or len(missing_keys):
            print("[W] Unexpecting Keys: ", unexpected_keys)
            print("[W] Missing Keys: ", missing_keys)

    model.to(device)
    if args.freeze_backbone:
        model.freeze_weights()

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank)

    lr = cfg.SOLVER.LR * args.num_gpus  # scale by num gpus
    optimizer = make_optimizer(cfg, model, lr)

    if len(cfg.SOLVER.LR_STEPS) > 1:
        milestones = [step // args.num_gpus for step in cfg.SOLVER.LR_STEPS]
    else:
        schedule_times = cfg.SOLVER.MAX_ITER // cfg.SOLVER.LR_STEPS[0]
        milestones = [cfg.SOLVER.LR_STEPS[0] //
                      args.num_gpus * i for i in range(1, schedule_times)]
    logger.info("Learning rate will be scheduled at {}".format(milestones))
    scheduler = make_lr_scheduler(cfg, optimizer, milestones)

    arguments = {"iteration": 0}
    save_to_disk = dist_util.get_rank() == 0
    checkpointer = CheckPointer(
        model, args.resume, optimizer, scheduler, cfg.OUTPUT_DIR, save_to_disk, logger)
    extra_checkpoint_data = checkpointer.load()
    arguments.update(extra_checkpoint_data)

    max_iter = cfg.SOLVER.MAX_ITER // args.num_gpus
    train_loader = make_data_loader(
        cfg, is_train=True, distributed=args.distributed, max_iter=max_iter, start_iter=arguments['iteration'])

    model = do_train(cfg, model, train_loader, optimizer,
                     scheduler, checkpointer, device, arguments, args)
    return model


def main():
    parser = argparse.ArgumentParser(
        description='Single Shot MultiBox Detector Training With PyTorch')
    parser.add_argument(
        "--config-file",
        default="./configs/resnet/dark/darknet_ssd320_mem_l.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--accumulation_steps",
        default=4,
        help="Gradient accumulation times",
        type=int,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--log_step', default=20, type=int,
                        help='Print logs every log_step')
    parser.add_argument('--save_step', default=1000, type=int,
                        help='Save checkpoint every save_step')
    parser.add_argument('--eval_step', default=1000, type=int,
                        help='Evaluate dataset every eval_step, disabled when eval_step < 0')
    parser.add_argument('--use_tensorboard', default=True, type=str2bool)
    parser.add_argument(
        "--resume", help="Pretrained checkpoint path", type=bool,
        default=False
    )
    parser.add_argument("--freeze-backbone", help="Whether freeze your backbone.",
                        default=False)
    parser.add_argument("--init-head", help="Whether freeze your backbone.",
                        default=True)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]
                   ) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    args.num_gpus = num_gpus

    if torch.cuda.is_available():
        # This flag allows you to enable the inbuilt cudnn auto-tuner to
        # find the best algorithm to use for your hardware.
        torch.backends.cudnn.benchmark = True
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://")
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if not cfg.OUTPUT_DIR:
        os.makedirs(cfg.OUTPUT_DIR)

    logger = setup_logger("SSD", dist_util.get_rank(), cfg.OUTPUT_DIR)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    model = train(cfg, args)

    if not args.skip_test:
        logger.info('Start evaluating...')
        torch.cuda.empty_cache()  # speed up evaluating after training finished
        do_evaluation(cfg, model, distributed=args.distributed)


if __name__ == '__main__':
    main()
