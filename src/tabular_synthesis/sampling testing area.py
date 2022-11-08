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


    dist_util.setup_dist()
    run = Run.get_context()
    # logger.configure()
    print_CUDA_information()

    print("Loading original data...")
    print(args.dataset_path)
    print(args.config_path)

    data, data_config = get_dataset(args.dataset_path, args.config_path)
    data = data.sample(n=400, random_state=42)

    print(data)

    tabular_loader = TabularLoader(
        data=data,
        test_ratio=0.0,
        patch_size=1,
        batch_size=args.batch_size,
        **data_config["dataset_config"])




    args.num_classes = tabular_loader.cond_generator_train.n_opt
    if args.num_samples <0:
        args.num_samples = abs(args.num_samples) * len(tabular_loader.data_train)
    args.in_channels = tabular_loader.patch_size 
    args.image_size = tabular_loader.side


    train_loader = TabularLoaderIterator(
        tabular_loader,
        num_iterations=args.num_samples//args.batch_size, 
        class_cond=args.class_cond)



    # noise_scalar = 1
    # noise =  th.randn(args.batch_size, args.image_size*args.image_size) * noise_scalar  # smaller noise maybe delete later
    # noise = tabular_loader.apply_activate(noise) # maybe delete later
    # noise = tabular_loader.image_transformer.transform(noise, padding="zero")
    # noise = noise.to(th.half).to(dist_util.dev())
    # noise =  th.randn(size=(args.batch_size, tabular_loader.patch_size, args.image_size, args.image_size)) * 0.1 # smaller noise maybe delete later
    # noise = noise.float().to(dist_util.dev())
    all_samples = []
    for i in range(args.num_samples // args.batch_size):
        batch, _ = next(train_loader)
        noise = th.rand_like(batch.float())*0.0
        sample = batch + noise
        sample = tabular_loader.inverse_batch(sample.cpu(), image_shape=True)
        original = tabular_loader.inverse_batch(batch.cpu(), image_shape=True)
        print(original.equals(sample))

        # gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        # dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_samples.append(sample)
    arr= pd.concat(all_samples)
        

    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        filename=f"{data_config['dataset_name']}_{shape_str}.csv"
        out_path = Path(args.output_path) / data_config['dataset_name'] 
        print(f"saving to {out_path}")
        # save as csv
        Path.mkdir(out_path, exist_ok=True)
        arr.to_csv(out_path / filename, index=False)

    dist.barrier()



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
    print("Found model options: ", options)
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
