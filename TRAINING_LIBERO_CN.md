# miniECoT 训练指南（中文，面向服务器）

本文档用于在 `embodied-CoT` 仓库中训练支持 mini 模型的 ECoT，重点覆盖 `libero_90 / libero_10`。

---

## 1. 前置准备

### 1.1 进入仓库

```bash
cd /path/to/miniECoT_v2/embodied-CoT
```

### 1.2 安装环境（示例）

```bash
conda create -n miniecot python=3.10 -y
conda activate miniecot

# 按你的 CUDA 平台安装 PyTorch（这里仅示例）
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

pip install -e .
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation
```

> 如果你后面还要跑 LIBERO 仿真评测，再额外安装：
> `pip install -r experiments/libero/libero_requirements.txt`

### 1.3 准备 HuggingFace Token

训练脚本默认会读取仓库根目录 `.hf_token`：

```bash
echo "hf_xxx你的token" > .hf_token
```

---

## 2. 数据准备

### 2.1 RLDS 数据根目录

你需要准备好 RLDS 格式的 LIBERO 数据，并记住根目录（后续用 `--data_root_dir` 指向它）。

本仓库当前支持的 LIBERO mixture key（训练常用）：
- `libero_90`
- `libero_10_no_noops`

### 2.2 reasoning 文件（ECoT 训练必备）

建议显式指定 reasoning 文件路径（最稳妥）：

```bash
export REASONING_DATASET_PATH=/abs/path/to/libero_reasonings.json
```

> 说明：
> - 代码已支持 LIBERO 的 step-level reasoning 格式（你给的 `libero_reasonings.json`）。
> - 若不设该环境变量，代码会尝试本地默认路径或 HuggingFace 下载。

---

## 3. 训练命令（mini 模型）

下面命令以 Qwen2.5 0.5B mini 配置为主。

先定义公共变量（推荐）：

```bash
export DATA_ROOT=/path/to/rlds_root
export RUN_ROOT=/path/to/train_runs
export WANDB_PROJECT=your_project
export WANDB_ENTITY=your_entity
```

### 3.1 单卡训练（推荐先跑通）

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/train.py \
  --vla.type prism-qwen25-dinosiglip-224px+0_5b+mx-libero-90 \
  --vla.expected_world_size 1 \
  --vla.global_batch_size 16 \
  --vla.per_device_batch_size 4 \
  --data_root_dir ${DATA_ROOT} \
  --run_root_dir ${RUN_ROOT} \
  --wandb_project ${WANDB_PROJECT} \
  --wandb_entity ${WANDB_ENTITY}
```

### 3.2 多卡训练（例如 8 卡）

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/train.py \
  --vla.type prism-qwen25-dinosiglip-224px+0_5b+mx-libero-90 \
  --data_root_dir ${DATA_ROOT} \
  --run_root_dir ${RUN_ROOT} \
  --wandb_project ${WANDB_PROJECT} \
  --wandb_entity ${WANDB_ENTITY}
```

### 3.3 训练不同 mini 配置

- **单帧主视角**：`prism-qwen25-dinosiglip-224px+0_5b+mx-libero-90`
- **历史帧 t=2**：`prism-qwen25-dinosiglip-224px-t2+0_5b+mx-libero-90`
- **wrist + 历史帧**：`prism-qwen25-dinosiglip-224px-wrist+0_5b+mx-libero-90`

只需替换 `--vla.type` 即可。

---

## 4. 在 libero_10 上训练

当前配置文件里已有 `libero_90` 的 mini 训练配置；训练 `libero_10` 时，直接覆盖 `data_mix`：

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/train.py \
  --vla.type prism-qwen25-dinosiglip-224px+0_5b+mx-libero-90 \
  --vla.data_mix libero_10_no_noops \
  --vla.expected_world_size 1 \
  --vla.global_batch_size 16 \
  --vla.per_device_batch_size 4 \
  --run_id qwen05b-libero10-ecot \
  --data_root_dir ${DATA_ROOT} \
  --run_root_dir ${RUN_ROOT} \
  --wandb_project ${WANDB_PROJECT} \
  --wandb_entity ${WANDB_ENTITY}
```

---

## 5. 断点续训

假设中断前 checkpoint 是：
`.../checkpoints/step-010000-epoch-20-loss=0.1234.pt`

则续训命令示例：

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/train.py \
  --vla.type prism-qwen25-dinosiglip-224px+0_5b+mx-libero-90 \
  --pretrained_checkpoint /abs/path/to/step-010000-epoch-20-loss=0.1234.pt \
  --is_resume True \
  --resume_step 10000 \
  --resume_epoch 20 \
  --vla.expected_world_size 1 \
  --vla.global_batch_size 16 \
  --vla.per_device_batch_size 4 \
  --data_root_dir ${DATA_ROOT} \
  --run_root_dir ${RUN_ROOT}
```

---

## 6. 训练输出与检查

- 运行目录：`${RUN_ROOT}/${run_id}/`
- 配置文件：`config.yaml`、`config.json`
- checkpoint：`${RUN_ROOT}/${run_id}/checkpoints/`
- 反归一化统计：`dataset_statistics.json`

建议先看两类日志是否正常：
- 是否成功读取 reasoning（LIBERO）
- `Action Token Accuracy` / loss 是否稳定下降

---

## 7. 训练后评测（可选）

你后面测 `libero_90 / libero_10` 成功率可用：
- `experiments/libero/run_libero_eval.py`

示例（libero_90）：

```bash
python experiments/libero/run_libero_eval.py \
  --model_family prismatic \
  --pretrained_checkpoint /abs/path/to/your_checkpoint.pt \
  --task_suite_name libero_90 \
  --num_trials_per_task 50
```
