"""Utilities for loading VLA models and querying actions during LIBERO evaluation."""

import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.load import load_vla

ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def _resolve_hf_token(token_ref):
    """Resolves HF token from file path or environment variable reference."""
    if token_ref is None:
        return None
    if isinstance(token_ref, Path):
        if token_ref.exists():
            return token_ref.read_text().strip()
        return None
    token_path = Path(token_ref)
    if token_path.exists():
        return token_path.read_text().strip()
    return os.environ.get(token_ref, None)


def set_seed_everywhere(seed: int) -> None:
    """Sets the random seed for Python, NumPy, and PyTorch functions."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_prismatic_vla(cfg):
    """Loads and returns a VLA model from checkpoint using Prismatic APIs."""
    print(f"[*] Initializing evaluation with model family `{cfg.model_family}`")
    hf_token = _resolve_hf_token(cfg.hf_token)
    print(f"[*] Loading VLA checkpoint from: {cfg.pretrained_checkpoint}")
    vla = load_vla(cfg.pretrained_checkpoint, hf_token=hf_token, load_for_training=False)
    for param in vla.parameters():
        assert param.dtype == torch.float32, f"Loaded VLM parameter not in full precision: {param}"

    vla.vision_backbone.to(dtype=vla.vision_backbone.half_precision_dtype)
    vla.llm_backbone.to(dtype=vla.llm_backbone.half_precision_dtype)
    vla.to(dtype=vla.llm_backbone.half_precision_dtype)
    vla.to(DEVICE)
    return vla


def get_vla(cfg):
    """Loads and returns an OpenVLA model from a Hugging Face checkpoint."""
    print("[*] Instantiating pretrained OpenVLA model")
    print("[*] Loading in BF16 with Flash Attention")

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.pretrained_checkpoint,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        load_in_8bit=cfg.load_in_8bit,
        load_in_4bit=cfg.load_in_4bit,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        vla = vla.to(DEVICE)

    dataset_statistics_path = os.path.join(cfg.pretrained_checkpoint, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            vla.norm_stats = json.load(f)

    return vla


def get_processor(cfg):
    """Gets Hugging Face processor for OpenVLA."""
    return AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)


def crop_and_resize(image, crop_scale, batch_size):
    """Center-crops image and resizes back to original dimensions."""
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [height_offsets, width_offsets, height_offsets + new_heights, width_offsets + new_widths],
        axis=1,
    )
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    if expanded_dims:
        image = image[0]
    return image


def apply_center_crop(im, t_h, t_w):
    """Applies a center crop to image."""
    assert im.shape[-3] >= t_h and im.shape[-2] >= t_w
    assert im.shape[-1] in [1, 3, 6]
    crop_h = int((im.shape[-3] - t_h) / 2)
    crop_w = int((im.shape[-2] - t_w) / 2)
    return im[..., crop_h : crop_h + t_h, crop_w : crop_w + t_w, :]


def get_vla_action(vla, processor, base_vla_name, obs, task_label, unnorm_key, center_crop=False):
    """Generates an action using OpenVLA."""
    if isinstance(obs["full_image"], list):
        obs["full_image"] = obs["full_image"][0]

    image = Image.fromarray(obs["full_image"]).convert("RGB")

    if center_crop:
        batch_size = 1
        crop_scale = 0.9
        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = crop_and_resize(image, crop_scale, batch_size)
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)
        image = Image.fromarray(image.numpy()).convert("RGB")

    if "openvla-v01" in base_vla_name:
        prompt = (
            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
        )
    else:
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)
    action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    return action


def get_prismatic_vla_action(
    vla, processor, base_vla_name, obs, task_label, unnorm_key, center_crop=False, **kwargs
):
    """Generates an action using Prismatic/OpenVLA model interface."""
    if not isinstance(obs["full_image"], list):
        obs["full_image"] = [obs["full_image"]]

    processed_images = []
    for img in obs["full_image"]:
        image = Image.fromarray(img).convert("RGB")

        if center_crop:
            temp_image = np.array(image)
            crop_scale = 0.9
            sqrt_crop_scale = math.sqrt(crop_scale)
            temp_image_cropped = apply_center_crop(
                temp_image,
                t_h=int(sqrt_crop_scale * temp_image.shape[0]),
                t_w=int(sqrt_crop_scale * temp_image.shape[1]),
            )
            temp_image = Image.fromarray(temp_image_cropped)
            temp_image = temp_image.resize(image.size, Image.Resampling.BILINEAR)
            image = temp_image

        processed_images.append(image)

    if len(processed_images) == 1:
        processed_images = processed_images[0]

    return vla.predict_action(processed_images, task_label, unnorm_key=unnorm_key, **kwargs)


def get_model(cfg):
    """Load model for evaluation."""
    if cfg.model_family in ["prismatic", "llava"]:
        model = get_prismatic_vla(cfg)
    elif cfg.model_family == "openvla":
        model = get_vla(cfg)
    else:
        raise ValueError(f"Unexpected `model_family`: {cfg.model_family}")
    print(f"Loaded model: {type(model)}")
    return model


def get_image_resize_size(cfg):
    """Gets image resize size for model class."""
    if cfg.model_family in ["prismatic", "openvla", "llava"]:
        return 224
    raise ValueError(f"Unexpected `model_family`: {cfg.model_family}")


def get_action(cfg, model, obs, task_label, processor=None):
    """Queries the model to get an action."""
    if cfg.model_family in ["prismatic", "llava"]:
        action = get_prismatic_vla_action(
            model, processor, cfg.pretrained_checkpoint, obs, task_label, cfg.unnorm_key, center_crop=cfg.center_crop
        )
    elif cfg.model_family == "openvla":
        action = get_vla_action(
            model, processor, cfg.pretrained_checkpoint, obs, task_label, cfg.unnorm_key, center_crop=cfg.center_crop
        )
    else:
        raise ValueError(f"Unexpected `model_family`: {cfg.model_family}")

    assert action.shape == (ACTION_DIM,)
    return action


def normalize_gripper_action(action, binarize=True):
    """Changes gripper action from [0, 1] to [-1, +1] (environment convention)."""
    action[..., -1] = 2 * (action[..., -1] - 0.0) / (1.0 - 0.0) - 1
    if binarize:
        action[..., -1] = np.sign(action[..., -1])
    return action


def invert_gripper_action(action):
    """Flips sign of gripper action to match simulator control semantics."""
    action[..., -1] = action[..., -1] * -1.0
    return action
