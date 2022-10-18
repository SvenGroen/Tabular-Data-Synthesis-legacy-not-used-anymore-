"""
Train a diffusion model on images.
"""

import argparse
from tabular_synthesis.synthesizer.loading.util import get_dataset

from tabular_synthesis.synthesizer.model.guided_diffusion import dist_util, logger
from tabular_synthesis.synthesizer.model.guided_diffusion.image_datasets import load_data
from tabular_synthesis.synthesizer.model.guided_diffusion.resample import create_named_schedule_sampler
from tabular_synthesis.synthesizer.model.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from tabular_synthesis.synthesizer.model.guided_diffusion.train_util import TrainLoop
from tabular_synthesis.synthesizer.loading.tabular_loader import TabularLoader, TabularLoaderIterator

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating data loader...")

    data, data_config = get_dataset(args.dataset_name)
    tabular_loader = TabularLoaderIterator(
        data=data,
        patch_size=1,
        num_iterations=args.iterations,
        batch_size=args.batch_size,
        **data_config["dataset_config"])

        # setting arguments for the classifier using the tabular_loader
    args.image_size = tabular_loader.side
    args.in_channels = tabular_loader.patch_size 
    args.out_channels = tabular_loader.patch_size * tabular_loader.cond_generator.n_opt

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)



    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=tabular_loader,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        dataset_name="",
        schedule_sampler="uniform",
        iterations=100,
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
