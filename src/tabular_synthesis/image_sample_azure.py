"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import pandas as pd
import torch.distributed as dist

from pathlib import Path

from tabular_synthesis.synthesizer.loading.util import get_dataset
from tabular_synthesis.synthesizer.loading.tabular_loader import TabularLoaderIterator, TabularLoader
from azureml.core import Run

from tabular_synthesis.synthesizer.model.guided_diffusion import dist_util, logger
from tabular_synthesis.synthesizer.model.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    model_name = select_model(startswith="ema", options=os.listdir(args.model_path))

    args.model_path = os.path.join(args.model_path, model_name)
    
    print("Trying to load: ", args.model_path)

    dist_util.setup_dist()
    run = Run.get_context()
    # logger.configure()
    print_CUDA_information()

    print("Loading original data...")
    print(args.dataset_path)
    print(args.config_path)

    data, data_config = get_dataset(args.dataset_path, args.config_path)
    tabular_loader = TabularLoader(
        data=data,
        test_ratio=0.2,
        patch_size=1,
        batch_size=args.batch_size,
        **data_config["dataset_config"])

    args.num_classes = tabular_loader.cond_generator_train.n_opt
    if args.num_samples == -1:
        args.num_samples = len(tabular_loader.data_train)
    args.in_channels = tabular_loader.patch_size 
    args.image_size = tabular_loader.side

    
    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())


    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    print("sampling...")
    all_samples = pd.DataFrame()
    all_labels = []
    while len(all_samples) < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=args.num_classes, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, tabular_loader.patch_size, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        # sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        # sample = sample.permute(0, 2, 3, 1)
        # sample = sample.contiguous()

        sample = tabular_loader.inverse_batch(sample.cpu(), image_shape=True)

        # gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        # dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_samples= pd.concat([all_samples, sample])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        print(f"created {len(all_samples)} samples")
    arr = all_samples.head(args.num_samples)

    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
        # to pandas series
        label_arr = pd.Series(label_arr)
        # add label column to arr
        arr["condition"] = label_arr
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        filename=f"{data_config['dataset_name']}_{shape_str}.csv"
        out_path = Path(args.output_path) / data_config['dataset_name'] 
        print(f"saving to {out_path}")
        # save as csv
        Path.mkdir(out_path, exist_ok=True)
        arr.to_csv(out_path / filename, index=False)

    dist.barrier()
    print("sampling complete")


def create_argparser():
    defaults = dict(
        dataset_path="",
        config_path="",
        model_path="",
        output_path="",
        class_cond=False,
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def select_model(startswith:str, options:list):
    options = [e for e in options if e.startswith(startswith)]
    options.sort()
    return options[-1]

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

if __name__ == "__main__":
    main()
