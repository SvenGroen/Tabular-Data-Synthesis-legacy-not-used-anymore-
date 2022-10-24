"""
Train a noised image classifier on ImageNet.
"""

import argparse
from email.mime import image
import os
import blobfile as bf
import joblib
import tempfile
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from azureml.core import Run

from tabular_synthesis.synthesizer.model.guided_diffusion import dist_util
from tabular_synthesis.synthesizer.model.guided_diffusion.fp16_util import MixedPrecisionTrainer
from tabular_synthesis.synthesizer.model.guided_diffusion.image_datasets import load_data
from tabular_synthesis.synthesizer.model.guided_diffusion.resample import create_named_schedule_sampler
from tabular_synthesis.synthesizer.model.guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    classifier_and_diffusion_defaults,
    create_classifier_and_diffusion,
)
from tabular_synthesis.synthesizer.model.guided_diffusion.train_util import parse_resume_step_from_filename, log_loss_dict
from tabular_synthesis.synthesizer.loading.util import get_dataset
from tabular_synthesis.synthesizer.loading.tabular_loader import TabularLoaderIterator, TabularLoader


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    run = Run.get_context()

    print_CUDA_information()
  
    data, data_config = get_dataset(args.dataset_path, args.config_path)
    data = data.sample(n=400, random_state=42)

    tabular_loader = TabularLoader(
        data=data,
        test_ratio=0.2,
        patch_size=1,
        batch_size=args.batch_size,
        **data_config["dataset_config"])



    train_loader = TabularLoaderIterator(tabular_loader, return_test=False, num_iterations=args.iterations)
    val_loader = TabularLoaderIterator(tabular_loader, return_test=True, num_iterations=args.iterations)


    # setting arguments for the classifier using the tabular_loader
    args.image_size = tabular_loader.side
    print(args.image_size)
    # args.image_size = -1
    print(tabular_loader.side)
    # args.classifier_resblock_updown = False
    args.in_channels = tabular_loader.patch_size 
    args.out_channels = tabular_loader.patch_size * tabular_loader.cond_generator_train.n_opt

    print("creating model and diffusion...")
    model, diffusion = create_classifier_and_diffusion(
        **args_to_dict(args, classifier_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    if args.noised:
        schedule_sampler = create_named_schedule_sampler(
            args.schedule_sampler, diffusion
        )

    resume_step = 0
    if args.resume_checkpoint:
        resume_step = parse_resume_step_from_filename(args.resume_checkpoint)
        if dist.get_rank() == 0:
            print(
                f"loading model from checkpoint: {args.resume_checkpoint}... at {resume_step} step"
            )
            model.load_state_dict(
                dist_util.load_state_dict(
                    args.resume_checkpoint, map_location=dist_util.dev()
                )
            )

    # Needed for creating correct EMAs and fp16 parameters.
    dist_util.sync_params(model.parameters())

    mp_trainer = MixedPrecisionTrainer(
        model=model, use_fp16=args.classifier_use_fp16, initial_lg_loss_scale=16.0
    )

    # print("Cuda_Usage before model creation: ", th.cuda.mem_get_info())
    model = DDP(
        model,
        device_ids= [dist_util.dev()] if th.cuda.is_available() else None,
        output_device= [dist_util.dev()] if th.cuda.is_available() else None,
        broadcast_buffers=False,
        bucket_cap_mb=128,
        find_unused_parameters=False,
    )
    # print("Cuda_Usage after model creation: ", th.cuda.mem_get_info())


    print(f"creating optimizer...")
    opt = AdamW(mp_trainer.master_params, lr=args.lr, weight_decay=args.weight_decay)
    if args.resume_checkpoint:
        opt_checkpoint = bf.join(
            bf.dirname(args.resume_checkpoint), f"opt{resume_step:06}.pt"
        )
        print(f"loading optimizer state from checkpoint: {opt_checkpoint}")
        opt.load_state_dict(
            dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev())
        )

    print("training classifier model...")

    def forward_backward_log(data_loader, prefix="train", log=True):
        # print("Cuda_Usage before batch loaded: ", th.cuda.mem_get_info())
        batch, extra = next(data_loader)
        labels = extra["y"].to(dist_util.dev())
        batch = batch.to(dist_util.dev())
        # print("Cuda_Usage after batch loaded: ", th.cuda.mem_get_info())

        # Noisy images
        if args.noised:
            t, _ = schedule_sampler.sample(batch.shape[0], dist_util.dev())
            batch = diffusion.q_sample(batch.float(), t) # <---transformed to float, maybe change back later
        else:
            t = th.zeros(batch.shape[0], dtype=th.long, device=dist_util.dev())

        for i, (sub_batch, sub_labels, sub_t) in enumerate(
            split_microbatches(args.microbatch, batch, labels, t)
        ):
            print(model.device)
            logits = model(sub_batch, timesteps=sub_t)
            logits2 = F.softmax(logits)
            # if batch.shape[1] != 1:
            #     logits = logits.view(logits.shape[0], batch.shape[1], -1)
            #     tmp_loss= []
            #     for i in range(batch.shape[1]):
            #         tmp_loss.append(
            #             F.cross_entropy(
            #                 logits[:, i, :],
            #                 sub_labels[:, i, :],
            #                 reduction="none",
            #             )
            #         )
            #     loss = th.stack(tmp_loss).mean(0)
            #     sub_labels = sub_labels.view(sub_labels.shape[0],-1).argmax(-1)
            #     logits = logits.mean(1)
            #     _test = F.cross_entropy(logits, sub_labels, reduction="none").detach()
                
            # else:
            sub_labels=sub_labels.squeeze(1).argmax(-1)
            loss =  F.cross_entropy(logits, sub_labels, reduction="none")
            loss_logits =  F.cross_entropy(logits2, sub_labels, reduction="none")
            _test=loss.detach()
            
            if log:
                run.log(f"{prefix}_loss", loss.detach().mean().item())
                run.log(f"{prefix}_acc@1", compute_top_k(
                    logits, sub_labels, k=1, reduction="none").mean().item())
                run.log(f"{prefix}_acc@5", compute_top_k(
                    logits, sub_labels, k=5, reduction="none").mean().item())
                # run.log(f"{prefix}_loss_diff", (loss.detach() - _test).mean().item())

                run.log(f"{prefix}_loss_softmax", loss_logits.detach().mean().item())
                run.log(f"{prefix}_acc@1_softmax", compute_top_k(
                    logits2, sub_labels, k=1, reduction="none").mean().item())
                run.log(f"{prefix}_acc@5_softmax", compute_top_k(
                    logits2, sub_labels, k=5, reduction="none").mean().item())

            loss = loss.mean()
            
            if loss.requires_grad:
                if i == 0:
                    mp_trainer.zero_grad()
                mp_trainer.backward(loss * len(sub_batch) / len(batch))
                # run.log("Cuda_Usage after backward", th.cuda.mem_get_info())

    for step in range(args.iterations - resume_step):
        if args.anneal_lr:
            set_annealed_lr(opt, args.lr, (step + resume_step) / args.iterations)
        log = (step + resume_step) % args.log_interval == 0
        if log:
            run.log("step", step + resume_step)
            run.log("samples",(step + resume_step + 1) * args.batch_size * dist.get_world_size())
            run.log("Lr", opt.param_groups[0]["lr"])
        forward_backward_log(data_loader=train_loader, log=log)
        mp_trainer.optimize(opt)
        if val_loader is not None and not step % args.eval_interval:
            with th.no_grad():
                with model.no_sync():
                    model.eval()
                    forward_backward_log(val_loader, prefix="val")
                    model.train()
        if (
            step
            and dist.get_rank() == 0
            and not (step + resume_step) % args.save_interval
        ):
            print("saving model...")
            save_model(mp_trainer, opt, step + resume_step, path=args.output_path)

    if dist.get_rank() == 0:
        print("saving model...")
        save_model(mp_trainer, opt, step + resume_step, path=args.output_path)
    dist.barrier()

    run.complete()


def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr

def print_CUDA_information():
    print("CUDA information:")
    if th.cuda.is_available():
        print(f"CUDA available: {th.cuda.is_available()}",
        f"CUDA version: {th.version.cuda}",
        f"Number of GPUs: {th.cuda.device_count()}",
        f"Current GPU: {th.cuda.current_device()}",
        f"Device name: {th.cuda.get_device_name(th.cuda.current_device())}",
        f"Device capability: {th.cuda.get_device_capability(th.cuda.current_device())}",
    )
    else:
        print("No CUDA available")

def save_model(mp_trainer, opt, step, path=None):
    if path is None:
        raise ValueError("Output path must be specified")
    if dist.get_rank() == 0:
        os.makedirs(path, exist_ok=True)
        th.save(
            mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
            os.path.join(f"{path}/model{step:06d}.pt"),
        )
        th.save(opt.state_dict(), os.path.join(f"{path}/opt{step:06d}.pt"))


def compute_top_k(logits, labels, k, reduction="mean"):
    _, top_ks = th.topk(logits, k, dim=-1)
    if reduction == "mean":
        return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()
    elif reduction == "none":
        return (top_ks == labels[:, None]).float().sum(dim=-1)


def split_microbatches(microbatch, *args):
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(x[i : i + microbatch] if x is not None else None for x in args)


def create_argparser():
    defaults = dict(
        dataset_path="src/tabular_synthesis/data/real_datasets/adult/adult.data",
        config_path="src/tabular_synthesis/data/config/adult.json",
        output_path="output",
        noised=True,
        iterations=150000,
        lr=3e-4,
        weight_decay=0.0,
        anneal_lr=False,
        batch_size=4,
        microbatch=-1,
        schedule_sampler="uniform",
        resume_checkpoint="",
        log_interval=10,
        eval_interval=5,
        save_interval=10000,
    )
    defaults.update(classifier_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
    
