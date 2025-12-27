import os
import random
import warnings

import wandb
import torch
import numpy as np

from model.note import NOTE
from dataloader.CIFAR10Dataset import CIFAR10Dataset
from parser import parse_arguments

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    args = parse_arguments()
    set_seed(args.seed)

    pretrain_config = {
        'device': args.device,
        'lr': args.src_lr,
        'weight_decay': args.src_weight_decay,
        'momentum': args.src_momentum,
        'batch_size': args.src_batch_size,
        'epochs': args.src_epochs,
        'iabn': args.iabn,
        'alpha': args.alpha,
        'memory_type': args.memory_type,
        'capacity': args.capacity,
        'conf_thresh': args.conf_thresh
    }

    online_config = {
        'epochs': args.tgt_epochs,
        'lr': args.tgt_lr,
        'weight_decay': args.tgt_weight_decay,
        'batch_size': args.tgt_batch_size,
        'use_learned_stats': args.tgt_use_learned_stats,
        'bn_momentum': args.tgt_bn_momentum,
        'update_interval': args.update_interval,
        'temp_factor': args.temp_factor,
        'optimize': args.optimize,
        'adapt': args.adapt,
        'weighted': args.weighted_loss,
        'e_margin': args.e_margin,
    }

    data_config = {
        'file_path': args.data_file_path,
        'distribution': args.distribution,
        'dir_beta': args.dir_beta,
        'shuffle_criterion': args.shuffle_criterion,
    }

    if args.corruption == 'all':
        # corruptions = ["shot_noise-5", "brightness-5", "motion_blur-5", "snow-5", "pixelate-5", "gaussian_noise-5", "defocus_blur-5",]
        #                #  "fog-5", "zoom_blur-5", "frost-5", "glass_blur-5", "impulse_noise-5", "contrast-5",
        #                # "jpeg_compression-5", "elastic_transform-5"]
        corruptions = ["shot_noise-5", "brightness-5", "motion_blur-5", "snow-5", "pixelate-5", "gaussian_noise-5", "defocus_blur-5",
                        "fog-5", "zoom_blur-5", "frost-5", "glass_blur-5", "impulse_noise-5", "contrast-5",
                        "jpeg_compression-5", "elastic_transform-5"]
    else:
        corruptions = [cor for cor in args.corruption.split()]
    
    wandb.login()
    run = wandb.init(
        project="myNOTE",
        config=args
    )
    run.config["note"] = "ent weight"

    source_dataset = CIFAR10Dataset(
        file_path=data_config["file_path"],
        domains=["original"],
        transform='src',
    )

    val_dataset = CIFAR10Dataset(
        file_path=data_config["file_path"],
        domains=["test"],
        transform='val',
    )

    ckpt_path = os.path.join(args.checkpoint_dir, args.method, str(args.seed))
    os.makedirs(ckpt_path, exist_ok=True)
    model = NOTE(source_dataset, val_dataset, checkpoint_path=ckpt_path, **pretrain_config)

    # Load checkpoint
    if not args.use_checkpoint:
        start_epoch = 0
    else:
        start_epoch = model.load_checkpoint() + 1
    print(f"Starting epoch {start_epoch}")

    # Source training
    best_acc, best_epoch = 0, 0
    for epoch in range(start_epoch, model.epochs):
        model.train()
        avg_loss, avg_acc = model.evaluation()

        if avg_acc > best_acc:
            best_acc = avg_acc
            best_epoch = epoch

        wandb.log({"accuracy": avg_acc, "loss": avg_loss})

        model.save_checkpoint(epoch)

    # Loaded fully-trained
    if best_acc == 0:
        _, best_acc = model.evaluation()
        best_epoch = model.epochs

    print(f'best_acc: {best_acc}, best_epoch: {best_epoch}')

    # Online train/test
    if not args.concat:
        for corruption in corruptions:
            target_dataset = CIFAR10Dataset(
                file_path=data_config["file_path"],
                domains=[corruption],
                transform='val',
                distribution=data_config["distribution"],
                dir_beta=data_config["dir_beta"],
                shuffle_criterion=data_config["shuffle_criterion"]
            )

            save_path = os.path.join(args.save_dir,
                                     args.method,
                                     data_config['distribution'],
                                     corruption,
                                     f"{args.model}_{'iabn' if args.iabn else 'bn'}_{args.shuffle_criterion}_conf{args.conf_thresh}_{'weight' if args.weighted_loss else 'none'}_{args.seed}")
            os.makedirs(save_path, exist_ok=True)
            print(f"Saving to: {save_path}")

            # Check checkpoint
            ckpt_exist = os.path.isfile(os.path.join(model.checkpoint_path, "pretrained_checkpoint.pth"))
            if not ckpt_exist:
                warnings.warn("WARNING: No pretrained checkpoint in online mode")
            # Reload model
            model = NOTE(source_dataset, val_dataset, checkpoint_path=ckpt_path, **pretrain_config) # Mostly dummy, to ensure proper reset only
            if ckpt_exist:
                model.load_checkpoint()
    
            model.setup_online(target_dataset=target_dataset, save_path=save_path, **online_config)
            for epoch in range(len(target_dataset)):
                model.train_online(epoch)

                if epoch > model.batch_size:
                    wandb.log({
                        f"{corruption}_online_accuracy": model.json["accuracy"][-1],
                        f"{corruption}_online_f1_macro": model.json["f1_macro"][-1],
                        f"{corruption}_memory_occupancy": model.mem.get_occupancy(),
                        f"{corruption}_online_confidence": np.mean(model.json['confidence']),
                    })

            model.dump_eval_online_result()
    else:
        target_dataset = CIFAR10Dataset(
            file_path=data_config["file_path"],
            domains=corruptions,
            transform='val',
            distribution=data_config["distribution"],
            dir_beta=data_config["dir_beta"],
            shuffle_criterion=data_config["shuffle_criterion"]
        )

        save_path = os.path.join(args.save_dir,
                                 args.method,
                                 data_config['distribution'],
                                 "multiple",
                                 f"{args.model}_{'iabn' if args.iabn else 'bn'}_{'dir' if args.distribution == 'dirichlet' else args.distribution}_{args.shuffle_criterion}")
        os.makedirs(save_path, exist_ok=True)
        print(f"Saving to: {save_path}")

        # Load/reset to checkpoint
        if not os.path.isfile(os.path.join(model.checkpoint_path, "pretrained_checkpoint.pth")):
            warnings.warn("WARNING: No pretrained checkpoint in online mode")
        model.load_checkpoint()

        model.setup_online(target_dataset=target_dataset, save_path=save_path, **online_config)
        for epoch in range(len(target_dataset)):
            model.train_online(epoch)

            if epoch > model.batch_size:
                wandb.log({
                    f"multiple_online_accuracy": model.json["accuracy"][-1],
                    f"multiple_online_f1_macro": model.json["f1_macro"][-1]
                })

        model.dump_eval_online_result()