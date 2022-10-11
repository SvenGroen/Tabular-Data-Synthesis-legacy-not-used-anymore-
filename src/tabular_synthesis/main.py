import os

from tabular_synthesis.synthesizer.loading.tabular_loader import TabularLoader, TabularLoaderIterator
from tabular_synthesis.synthesizer.model.improved_diffusion import script_util, dist_util, logger
from tabular_synthesis.synthesizer.model.improved_diffusion.resample import create_named_schedule_sampler
from tabular_synthesis.synthesizer.model.improved_diffusion.train_util import TrainLoop
import pandas as pd
import torch as th
import torch.distributed as dist
import numpy as np

dataset = "Adult"
# Specifying the path of the dataset used
real_path = "data/real_datasets/adult/mini_adult.csv"

test_ratio = 0.20
categorical_columns = ['workclass', 'education', 'marital-status', 'occupation',
                       'relationship', 'race', 'gender', 'native-country', 'income']
log_columns = []
mixed_columns = {'capital-loss': [0.0], 'capital-gain': [0.0]}
integer_columns = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']
problem_type = {"Classification": 'income'}
model_path = "C:\\Users\\SvenG\\AppData\\Local\\Temp\\openai-2022-10-10-15-42-06-642134\\opt000009.pt"
epochs = 1

if __name__ == '__main__':
    data = pd.read_csv(real_path)
    logger.configure()
    init_arguments = {"test_ratio": 0.20,
                      "categorical_columns": ['workclass', 'education', 'marital-status', 'occupation', 'relationship',
                                              'race', 'gender', 'native-country', 'income'],
                      "log_columns": [],
                      "mixed_columns": {'capital-loss': [0.0],
                                        'capital-gain': [0.0]},
                      "general_columns": ["age"],
                      "non_categorical_columns": [],
                      "integer_columns": ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week'],
                      "batch_size": 32
                      }

    tabular_loader = TabularLoaderIterator(data=data, num_iterations=10, **init_arguments)
    dist_util.setup_dist()
    batch, cond, col, opt = tabular_loader.get_batch(image_shape=True)
    model, diffusion = script_util.create_model_and_diffusion(
        image_size=tabular_loader.side,
        class_cond=False,
        learn_sigma=True,
        sigma_small=False,
        num_channels=128,
        num_res_blocks=3,
        num_heads=4,
        num_heads_upsample=-1,
        attention_resolutions="16,8",
        dropout=0.0,
        diffusion_steps=50,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=True,
        use_checkpoint=False,
        use_scale_shift_norm=True,
    )
    schedule_sampler = create_named_schedule_sampler("uniform", diffusion)
    # set cpu device
    # device = th.device("cpu")
    # model.to(dist_util.dev())

    # TrainLoop(
    #     model=model,
    #     diffusion=diffusion,
    #     data=tabular_loader,
    #     batch_size=init_arguments["batch_size"],
    #     microbatch=1,
    #     lr=1e-4,
    #     ema_rate="0.9999",
    #     log_interval=10,
    #     save_interval=1,
    #     resume_checkpoint="",
    #     use_fp16=False,
    #     fp16_scale_growth=0.001,
    #     schedule_sampler=schedule_sampler,
    #     weight_decay=0.0,
    #     lr_anneal_steps=0,
    # ).run_loop()

    # sample
    # model.load_state_dict(dist_util.load_state_dict(model_path, map_location="cpu"))
    model.to(dist_util.dev())
    model.eval()

    num_samples = 1
    NUM_CLASSES = len(cond)
    use_ddim = False
    class_cond = False
    clip_denoised = True

    all_images = []
    all_labels = []
    while len(all_images) * init_arguments["batch_size"] < num_samples:
        model_kwargs = {}
        if class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(init_arguments["batch_size"],), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (init_arguments["batch_size"], 1, tabular_loader.side, tabular_loader.side),  # bs, channels, height, width
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
        )

        inverse_batch = tabular_loader.inverse_batch(sample, image_shape=True)
        # sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)  # ((sample +1 )* 0.5).clamp(0, 1)?
        # sample = sample.permute(0, 2, 3, 1)
        # sample = sample.contiguous()
        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: num_samples]

    # inverse_batch = tabular_loader.inverse_batch(arr)
    # plt.imshow(tmp)
    # plt.show()

    if class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")
