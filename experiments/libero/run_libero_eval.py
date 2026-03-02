"""
run_libero_eval.py

Runs a VLA model in LIBERO simulation and reports per-task/overall success rates.

Example:
    python experiments/libero/run_libero_eval.py \
        --model_family prismatic \
        --pretrained_checkpoint <CHECKPOINT_PT> \
        --task_suite_name libero_90 \
        --num_trials_per_task 50 \
        --obs_history 1 \
        --use_wrist_image False
"""

import json
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.libero.model_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    get_processor,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclass
class GenerateConfig:
    # Model
    model_family: str = "prismatic"  # [prismatic | llava | openvla]
    hf_token: Union[str, Path] = Path(".hf_token")
    pretrained_checkpoint: Union[str, Path] = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True
    obs_history: int = 1
    use_wrist_image: bool = False

    # LIBERO
    task_suite_name: str = "libero_90"  # [libero_spatial | libero_object | libero_goal | libero_10 | libero_90]
    num_steps_wait: int = 10
    num_trials_per_task: int = 50

    # Logging
    save_videos: bool = True
    video_dir: str = "./experiments/rollouts"
    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    prefix: str = ""

    # W&B (optional)
    use_wandb: bool = False
    wandb_project: str = "prismatic"
    wandb_entity: Optional[str] = None

    # Reproducibility
    seed: int = 7


def _validate_and_prepare(cfg: GenerateConfig, model) -> None:
    if not cfg.pretrained_checkpoint:
        raise ValueError("`pretrained_checkpoint` must be set.")
    if cfg.task_suite_name not in SUITE_MAX_STEPS:
        raise ValueError(f"Unsupported `task_suite_name`: {cfg.task_suite_name}")
    if cfg.obs_history < 1:
        raise ValueError("`obs_history` must be >= 1.")
    if cfg.load_in_8bit and cfg.load_in_4bit:
        raise ValueError("Cannot enable both 8-bit and 4-bit quantization.")
    if "image_aug" in str(cfg.pretrained_checkpoint) and not cfg.center_crop:
        raise ValueError("Checkpoint looks image-augmented, please set `center_crop=True`.")

    cfg.unnorm_key = cfg.task_suite_name
    if cfg.model_family in ["openvla", "prismatic", "llava"]:
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        if cfg.unnorm_key not in model.norm_stats:
            raise KeyError(
                f"Action un-normalization key `{cfg.unnorm_key}` not found in checkpoint `norm_stats`. "
                f"Available keys include: {list(model.norm_stats.keys())[:10]}"
            )


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    set_seed_everywhere(cfg.seed)

    model = get_model(cfg)
    _validate_and_prepare(cfg, model)

    processor = get_processor(cfg) if cfg.model_family == "openvla" else None
    resize_size = get_image_resize_size(cfg)
    max_steps = SUITE_MAX_STEPS[cfg.task_suite_name]

    run_id = f"{cfg.prefix}EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note:
        run_id += f"--{cfg.run_id_note}"

    os.makedirs(cfg.local_log_dir, exist_ok=True)
    log_path = os.path.join(cfg.local_log_dir, f"{run_id}.txt")
    summary_path = os.path.join(cfg.local_log_dir, f"{run_id}.json")
    log_file = open(log_path, "w", encoding="utf-8")
    print(f"Logging to {log_path}")

    wandb = None
    if cfg.use_wandb:
        import wandb as wandb_lib

        wandb = wandb_lib
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=run_id)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name} | tasks={num_tasks}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    total_episodes = 0
    total_successes = 0
    per_task_metrics = []

    for task_id in tqdm.tqdm(range(num_tasks), desc="Tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, resolution=resize_size)

        task_episodes = 0
        task_successes = 0

        if len(initial_states) == 0:
            raise RuntimeError(f"No initial states for task_id={task_id}")

        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task), desc=f"Task {task_id}", leave=False):
            print(f"\nTask[{task_id}] {task_description}")
            log_file.write(f"\nTask[{task_id}] {task_description}\n")

            env.reset()
            state_idx = episode_idx % len(initial_states)
            obs = env.set_init_state(initial_states[state_idx])

            t = 0
            done = False
            replay_images = []
            replay_wrist_images = []

            while t < max_steps + cfg.num_steps_wait:
                try:
                    if t < cfg.num_steps_wait:
                        obs, _, _, _ = env.step(get_libero_dummy_action())
                        t += 1
                        continue

                    img = get_libero_image(obs, resize_size)
                    replay_images.append(img)

                    if cfg.use_wrist_image:
                        wrist_img = get_libero_image(obs, resize_size, key="robot0_eye_in_hand_image")
                        replay_wrist_images.append(wrist_img)

                    image_history = replay_images[-cfg.obs_history :]
                    if len(image_history) < cfg.obs_history:
                        image_history.extend([replay_images[-1]] * (cfg.obs_history - len(image_history)))

                    if cfg.use_wrist_image:
                        wrist_history = replay_wrist_images[-cfg.obs_history :]
                        if len(wrist_history) < cfg.obs_history:
                            wrist_history.extend([replay_wrist_images[-1]] * (cfg.obs_history - len(wrist_history)))
                        image_history = [val for pair in zip(image_history, wrist_history) for val in pair]

                    observation = {
                        "full_image": image_history,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }

                    action = get_action(cfg, model, observation, task_description, processor=processor)
                    action = normalize_gripper_action(action, binarize=True)
                    if cfg.model_family in ["openvla", "prismatic", "llava"]:
                        action = invert_gripper_action(action)

                    obs, _, done, _ = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break

                    t += 1
                except Exception as exc:
                    print(f"Caught exception in rollout: {exc}")
                    log_file.write(f"Caught exception in rollout: {exc}\n")
                    traceback.print_exc()
                    done = False
                    break

            task_episodes += 1
            total_episodes += 1

            if cfg.save_videos and len(replay_images) > 0:
                save_rollout_video(
                    replay_images,
                    idx=total_episodes,
                    success=done,
                    task_description=task_description,
                    output_root=cfg.video_dir,
                    log_file=log_file,
                )

            success_rate = total_successes / total_episodes
            print(f"Episode done={done} | total={total_episodes} | successes={total_successes} ({success_rate * 100:.1f}%)")
            log_file.write(
                f"Episode done={done} | total={total_episodes} | successes={total_successes} ({success_rate * 100:.1f}%)\n"
            )
            log_file.flush()

            if wandb is not None:
                wandb.log({"success_rate/total_online": success_rate, "num_episodes/total_online": total_episodes})

        env.close()

        task_rate = task_successes / max(task_episodes, 1)
        total_rate = total_successes / max(total_episodes, 1)
        per_task_metrics.append(
            {
                "task_id": task_id,
                "task_description": task_description,
                "episodes": task_episodes,
                "successes": task_successes,
                "success_rate": task_rate,
            }
        )

        print(f"Task[{task_id}] success_rate={task_rate:.4f} | total_success_rate={total_rate:.4f}")
        log_file.write(f"Task[{task_id}] success_rate={task_rate:.4f} | total_success_rate={total_rate:.4f}\n")
        log_file.flush()

        if wandb is not None:
            wandb.log({f"success_rate/task_{task_id}": task_rate, f"num_episodes/task_{task_id}": task_episodes})

    summary = {
        "run_id": run_id,
        "task_suite_name": cfg.task_suite_name,
        "model_family": cfg.model_family,
        "pretrained_checkpoint": str(cfg.pretrained_checkpoint),
        "num_tasks": num_tasks,
        "num_trials_per_task": cfg.num_trials_per_task,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "total_success_rate": total_successes / max(total_episodes, 1),
        "per_task": per_task_metrics,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved summary JSON to {summary_path}")

    log_file.close()
    if wandb is not None:
        wandb.log({"success_rate/total": summary["total_success_rate"], "num_episodes/total": total_episodes})
        wandb.save(log_path)
        wandb.save(summary_path)


if __name__ == "__main__":
    eval_libero()
