import collections
import datetime
import logging
import os
import time
import torch
import torch.distributed as dist

# from ssd.engine.inference import do_evaluation
from val import do_evaluation
from ssd.utils import dist_util
from ssd.utils.metric_logger import MetricLogger
from inference.ssd_xmem_infer_detector import SSDXMemInference


def write_metric(eval_result, prefix, summary_writer, global_step):
    for key in eval_result:
        value = eval_result[key]
        tag = '{}/{}'.format(prefix, key)
        if isinstance(value, collections.Mapping):
            write_metric(value, tag, summary_writer, global_step)
        else:
            summary_writer.add_scalar(tag, value, global_step=global_step)


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = dist_util.get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(cfg, model,
             data_loader,
             optimizer,
             scheduler,
             checkpointer,
             device,
             arguments,
             args):
    logger = logging.getLogger("SSD.trainer")
    logger.info("Start training ...")
    meters = MetricLogger()

    model.train()
    if args.freeze_backbone:
        model.freeze_weights()

    save_to_disk = dist_util.get_rank() == 0
    if args.use_tensorboard and save_to_disk:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            from tensorboardX import SummaryWriter
        summary_writer = SummaryWriter(log_dir=os.path.join(cfg.OUTPUT_DIR, 'tf_logs'))
    else:
        summary_writer = None

    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    if "best_map" in arguments:
        best_map = arguments["best_map"]
    else:
        best_map = 0.0
    start_training_time = time.time()
    end = time.time()

    for iteration, (data, targets) in enumerate(data_loader, start_iter):
        arguments["iteration"] = iteration
        arguments["best_map"] = best_map

        curr_images, prev_images = data["curr"], data["prev"]
        curr_images, prev_images = curr_images.to(device), prev_images.to(device)
        targets = targets.to(device)
        if curr_images.shape[0] == 1:
            continue

        iteration = iteration + 1
        if args.freeze_backbone:
            model.freeze_weights()
        loss_dict = model(prev_images, curr_images, targets=targets)

        # 单次梯度计算并更新
        if int(args.accumulation_steps) <= 1:

            loss = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        # N次梯度累积并更新
        else:
            loss = sum(loss for loss in loss_dict.values()) / float(args.accumulation_steps)
            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(total_loss=losses_reduced, **loss_dict_reduced)

            loss.backward()
            if ((iteration) % args.accumulation_steps) == 0:
                # optimizer the net
                optimizer.step()  # update parameters of net
                optimizer.zero_grad()
            scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time)
        if iteration % args.log_step == 0:
            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            if device == "cuda":
                logger.info(
                    meters.delimiter.join([
                        "iter: {iter:06d}",
                        "lr: {lr:.8f}",
                        '{meters}',
                        "eta: {eta}",
                        'mem: {mem}M',
                    ]).format(
                        iter=iteration,
                        lr=optimizer.param_groups[0]['lr'],
                        meters=str(meters),
                        eta=eta_string,
                        mem=round(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0),
                    )
                )
            else:
                logger.info(
                    meters.delimiter.join([
                        "iter: {iter:06d}",
                        "lr: {lr:.8f}",
                        '{meters}',
                        "eta: {eta}",
                    ]).format(
                        iter=iteration,
                        lr=optimizer.param_groups[0]['lr'],
                        meters=str(meters),
                        eta=eta_string,
                    )
                )
            if summary_writer:
                global_step = iteration
                summary_writer.add_scalar('losses/total_loss', losses_reduced, global_step=global_step)
                for loss_name, loss_item in loss_dict_reduced.items():
                    summary_writer.add_scalar('losses/{}'.format(loss_name), loss_item, global_step=global_step)
                summary_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)

        if iteration % args.save_step == 0:
            # checkpointer.save("model_{:06d}".format(iteration), **arguments)
            checkpointer.save("last_iteration_model", **arguments)

        if args.eval_step > 0 and iteration % args.eval_step == 0 and not iteration == max_iter:
            checkpointer.save("last_iteration_model", **arguments)
            ckpt = "{}/last_iteration_model.pth".format(checkpointer.save_dir)
            device = torch.device(cfg.MODEL.DEVICE)
            # inference_session = SSDXMemInference(cfg=cfg, weights=ckpt, device=device, score_th=0.5)
            # mAP = do_evaluation(cfg, inference_session, distributed=args.distributed, iteration=iteration)
            model.to(device)
            checkpointer.load(ckpt, use_latest=ckpt is None)
            mAP = do_evaluation(cfg, model.eval(), distributed=args.distributed, iteration=iteration)
            # if dist_util.get_rank() == 0 and summary_writer:
            #     for eval_result, dataset in zip(eval_results, cfg.DATASETS.TEST):
            #         write_metric(eval_result['metrics'], 'metrics/' + dataset, summary_writer, iteration)
            if mAP > best_map:
                logger.info("New best mAP ({:.3f}) model".format(mAP))
                best_map = mAP
                checkpointer.save("best_map_model_{}".format(int(best_map * 100)))
                logger.info("New best mAP ({:.3f}) model has been saved to best_map_model.pth".format(mAP))
            # *IMPORTANT*: change to train mode after eval.
            model.train()
        # model.detach_hidden()

    checkpointer.save("model_final", **arguments)
    # compute training time
    total_training_time = int(time.time() - start_training_time)
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total training time: {} ({:.4f} s / it)".format(total_time_str, total_training_time / max_iter))
    return model
