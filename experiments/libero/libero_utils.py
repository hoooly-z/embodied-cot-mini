"""Utilities for evaluating VLA policies in LIBERO simulation environments."""

import math
import os
import time

import imageio
import numpy as np
import tensorflow as tf
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image

DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")


def get_libero_env(task, resolution=256):
    """Initializes a LIBERO task environment and returns (env, task_description)."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # Affects object placements even with fixed init state.
    return env, task_description


def get_libero_dummy_action():
    """Returns a no-op action used during initial stabilization steps."""
    return [0, 0, 0, 0, 0, 0, -1]


def resize_image(img, resize_size):
    """
    Resizes an image with the same JPEG encode/decode path used by RLDS preprocessing.
    """
    assert isinstance(resize_size, tuple)
    img = tf.image.encode_jpeg(img)
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    return img.numpy()


def get_libero_image(obs, resize_size, key="agentview_image"):
    """Extracts and preprocesses one camera view from LIBERO observation."""
    assert isinstance(resize_size, (int, tuple))
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = np.flipud(obs[key])
    img = Image.fromarray(img)
    img = img.resize(resize_size, Image.Resampling.LANCZOS)
    img = img.convert("RGB")
    return np.array(img)


def save_rollout_video(rollout_images, idx, success, task_description, output_root="./rollouts", log_file=None):
    """Saves an MP4 replay of one evaluation episode and returns the path."""
    rollout_dir = os.path.join(output_root, DATE)
    os.makedirs(rollout_dir, exist_ok=True)
    task_tag = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = os.path.join(
        rollout_dir,
        f"{DATE_TIME}--episode={idx}--success={success}--task={task_tag}.mp4",
    )

    writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        writer.append_data(img)
    writer.close()

    msg = f"Saved rollout MP4 at path {mp4_path}"
    print(msg)
    if log_file is not None:
        log_file.write(msg + "\n")
    return mp4_path


def quat2axisangle(quat):
    """
    Converts quaternion (x, y, z, w) into axis-angle exponential coordinates.
    """
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den
