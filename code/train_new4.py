import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path
import pdb
import itertools
import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
# from datasets import load_dataset
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig, CLIPImageProcessor

import diffusers
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from pipeline_sd15_modify_new import StableDiffusionControlNetPipeline
from diffusers import UNet2DConditionModel as OriginalUNet2DConditionModel
# from utils.unet_with_adapter import UNet2DConditionModel as UNet2DConditionModelWithAdapter
from fusion_model.encoder_plus_new import fusion_model_v1, fusion_model_v2
from albumentations.pytorch import ToTensorV2
import albumentations as A
import cv2
from safetensors.torch import load_file

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.23.0")

logger = get_logger(__name__)

def concatenate_images(image_files, output_file):
    images = image_files  # list
    max_height = max(img.height for img in images)
    images = [img.resize((img.width, max_height)) for img in images]
    total_width = sum(img.width for img in images)
    combined = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for img in images:
        combined.paste(img, (x_offset, 0))
        x_offset += img.width
    combined.save(output_file)

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def log_validation(vae, text_encoder, tokenizer, unet, controlnet_background, foreground_encoder, args, accelerator, weight_dtype, step):
    logger.info("Running validation... ")
    controlnet_background = accelerator.unwrap_model(controlnet_background)
    foreground_encoder = accelerator.unwrap_model(foreground_encoder)
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet_background,
        safety_checker=None,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    validation_backgrounds = args.validation_backgrounds
    validation_foregrounds = args.validation_foregrounds
    image_logs = []
    validation_path = os.path.join(args.output_dir, "validation", f"step-{step}")
    os.makedirs(validation_path, exist_ok=True)
    _num = 0
    for validation_background, validation_foreground in zip(validation_backgrounds, validation_foregrounds):
        _num += 1
        validation_background = Image.open(validation_background).convert("RGB").resize((512, 512))
        validation_foreground = Image.open(validation_foreground).convert("RGB").resize((512, 512))
        images = []
        for num in range(args.num_validation_images):
            with torch.autocast("cuda"):
                image = foreground_encoder.generate_modify_new(background_image=validation_background, foreground_image=validation_foreground, pipe=pipeline)
                concatenate_images([validation_background, validation_foreground, image], os.path.join(validation_path, str(num)+str(_num)+".png"))
            images.append(image)
        image_logs.append(
            {"validation_background": validation_background, "validation_foreground": validation_foreground, "validation_results": images}
        )


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_image.save(os.path.join(repo_folder, "image_control.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_image] + images
            image_grid(images, 1, len(images)).save(os.path.join(repo_folder, f"images_{i}.png"))
            img_str += f"![images_{i})](./images_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- controlnet
inference: true
---
    """
    model_card = f"""
# controlnet-{repo_id}

These are controlnet weights trained on {base_model} with new type of conditioning.
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models."
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=300)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=10000,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint1",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="./logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--validation_backgrounds",
        type=str,
        default=["./background/1.png"],
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_foregrounds",
        type=str,
        default=["./foreground/1.png"],
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=5000,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_controlnet",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args


class MyDataset(Dataset):    
    def __init__(self, root):
        """
        get images and execute transforms.
        """
        background_imgs = [os.path.join(root, img) for img in os.listdir(root)]

        background_imgs = sorted(background_imgs, key=lambda x: x.split('.')[-2])
        self.background_imgs = background_imgs

        prob = 0.5
        OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
        OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

        self.clip_norm = transforms.Normalize(OPENAI_CLIP_MEAN, OPENAI_CLIP_STD)
        self.norm = transforms.Normalize([0.5], [0.5])
        self.to_tensor = transforms.ToTensor()

        self.pixel_transform = A.Compose([
            A.SmallestMaxSize(max_size=512),
            A.CenterCrop(512, 512),
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=None, translate_percent=(-0.1, 0.1), rotate=(-10, 10), p=0.5),
        ], additional_targets={'image0': 'image', 'image1': 'image'})

        self.clip_transform = A.Compose([
            A.SmallestMaxSize(max_size=224),
            A.CenterCrop(224, 224),
            A.Affine(scale=(0.7, 1.3), translate_percent=(-0.1, 0.1), rotate=(-25, 25), p=0.7),
            A.OneOf(
                [
                    A.PixelDropout(dropout_prob=0.1, p=prob),
                    A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=prob),
                    A.GaussianBlur(blur_limit=(3, 5), p=prob),
                ]
            )]
        )

    def clip_imgaug(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.clip_transform(image=image)
        image = self.clip_norm(self.to_tensor(results["image"]/255.))
        return image

    def imgaug(self, background_image, gt_image):
        background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        results = self.pixel_transform(image=background_image, image0=gt_image)
        background_image, gt_image = self.to_tensor(results["image"]/255.), self.norm(self.to_tensor(results["image0"]/255.))
        return background_image, gt_image
                       
    def __getitem__(self, index):
        """
        return data
        """
        background_path = self.background_imgs[index]
        gt_path = background_path.replace('background', 'gt')
        foreground_path = background_path.replace('background', 'foreground')

        background_data = cv2.imread(background_path)
        h, w, c = background_data.shape
        gt_data = cv2.imread(gt_path)
        gt_data = cv2.resize(gt_data,(w,h))
        background_data, gt_data = self.imgaug(background_data, gt_data)

        foreground_data = cv2.imread(foreground_path)
        foreground_data = cv2.resize(foreground_data,(w,h))
        foreground_data = self.clip_imgaug(foreground_data)
        return background_data, gt_data, foreground_data
  
    def __len__(self):
        """
        return images size.
        """
        return len(self.background_imgs)


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    if args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = OriginalUNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    if args.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights")
        controlnet_background = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
    else:
        logger.info("Initializing controlnet weights from unet")
        controlnet_background = ControlNetModel.from_unet(unet)

    image_encoder_path = "/home/hadoop-kg-llm-ddpt/dolphinfs_hdd_hadoop-kg-llm-ddpt/Foodfusion/models/image_encoder_l"
    print("===============Aply fusion_model_v2 SD image!=================")
    # foreground_encoder = fusion_model_v1(unet, image_encoder_path, accelerator.device, dtype=torch.float32)
    foreground_encoder = fusion_model_v2(unet, image_encoder_path, accelerator.device, dtype=torch.float32)
    # =================================================================================================

    if args.resume_from_checkpoint is not None and args.resume_from_checkpoint1 is not None:
        print("Load foreground_encoder: %s" %args.resume_from_checkpoint)
        print("Load background_encoder: %s" %args.resume_from_checkpoint1)
        foreground_state_dict = load_file(args.resume_from_checkpoint, device="cuda")
        background_state_dict = load_file(args.resume_from_checkpoint1, device="cuda")
        controlnet_background.load_state_dict(background_state_dict, strict=False)
        foreground_encoder.load_state_dict(foreground_state_dict, strict=False)
        del foreground_state_dict
        del background_state_dict
        controlnet_background.to(accelerator.device)
        foreground_encoder.to(accelerator.device)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    controlnet_background.requires_grad_(True)

    foreground_encoder.SSR_layers.requires_grad_(True)
    foreground_encoder.resampler.requires_grad_(True)
    foreground_encoder.image_encoder.requires_grad_(False)
    # ======================================================

    optimizer_class = torch.optim.AdamW

    params_to_optimize = itertools.chain(controlnet_background.parameters(),
                                        foreground_encoder.SSR_layers.parameters(),
                                        foreground_encoder.resampler.parameters())

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_data_root = '/home/hadoop-kg-llm-ddpt/dolphinfs_hdd_hadoop-kg-llm-ddpt/shichaohua123/Food_more_select/background/'
    train_dataset = MyDataset(train_data_root)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    foreground_encoder, controlnet_background, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        foreground_encoder, controlnet_background, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_foregrounds")
        tracker_config.pop("validation_backgrounds")
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Train data root = {train_data_root}")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Save checkpoints steps = {args.checkpointing_steps}")
    logger.info(f"  Save validation steps = {args.validation_steps}")
    logger.info(f"  Output paths = {args.output_dir}")
    logger.info(f"  Input Text = Null-text")
    global_step = 0
    first_epoch = 0
    initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    null_text_inputs = tokenizer(
        "", max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids
    # Get the text embedding for conditioning
    encoder_hidden_states = text_encoder(null_text_inputs.to(device=accelerator.device))[0]

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, (background, gt, foreground) in enumerate(train_dataloader):
            with accelerator.accumulate(foreground_encoder):

                # Convert images to latent space
                latents = vae.encode(gt.to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                controlnet_image1 = background.to(dtype=weight_dtype)
                down_block_res_samples_background, mid_block_res_sample_background = controlnet_background(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states.repeat(bsz, 1, 1),
                    controlnet_cond=controlnet_image1,
                    return_dict=False,
                )

                down_block_res_samples = [samples_prev for samples_prev in down_block_res_samples_background]
                mid_block_res_sample = mid_block_res_sample_background

                image_embeds1 = foreground_encoder(foreground.to(accelerator.device, dtype=weight_dtype))
                background_resized = F.interpolate(background, size=(224, 224), mode='bilinear', align_corners=False)
                image_embeds2 = foreground_encoder(background_resized.to(accelerator.device, dtype=weight_dtype))  # [bsz, 3084, 1024]
                image_embeds = torch.cat((image_embeds1, image_embeds2), 1) 

                # Predict the noise residual
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=image_embeds,
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                ).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if args.validation_backgrounds is not None and global_step % args.validation_steps == 0:
                        log_validation(
                            vae,
                            text_encoder,
                            tokenizer,
                            unet,
                            controlnet_background,
                            foreground_encoder,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                        )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
