# Flow-GRPO / ExploreNet training on Stable Diffusion 3.5 Medium.
#
# Derived from the Flow-GRPO SD3 training script
# (https://github.com/yifan123/flow_grpo, MIT License, Copyright (c) 2025 Jie Liu).
# ExploreNet adds a learned noise policy (a second SD3 transformer with LoRA)
# that sets the per-element SDE exploration noise and is trained jointly with
# the denoiser. With `config.sample.use_learned_noise_policy = False` this
# script runs plain Flow-GRPO.
from collections import defaultdict
import contextlib
import os


def _setup_node_local_cache_env():
    """Keep Triton/TorchInductor cache writes on node-local temporary storage."""
    local_root = os.environ.get("TMPDIR") or "/tmp"
    user = os.environ.get("USER", "user")
    cache_root = os.path.join(local_root, user, "explorenet_cache")
    for env_name, subdir in (
        ("TRITON_CACHE_DIR", "triton"),
        ("TORCHINDUCTOR_CACHE_DIR", "torchinductor"),
        ("XDG_CACHE_HOME", "xdg"),
    ):
        os.environ.setdefault(env_name, os.path.join(cache_root, subdir))
        os.makedirs(os.environ[env_name], exist_ok=True)


_setup_node_local_cache_env()

import datetime
from concurrent import futures
import time
import json
import hashlib
from absl import app, flags
from accelerate import Accelerator, InitProcessGroupKwargs
from ml_collections import config_flags
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from accelerate.logging import get_logger
from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel
from diffusers.utils.torch_utils import is_compiled_module
import numpy as np
import torch
import torch.distributed as dist
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, PeftModel
import random
from torch.utils.data import DataLoader

from explorenet.data import GenevalPromptDataset, DistributedKRepeatSampler
import explorenet.rewards
from explorenet.stat_tracking import PerPromptStatTracker
from explorenet.diffusers_patch.sd3_pipeline_with_logprob import pipeline_with_logprob
from explorenet.diffusers_patch.sd3_sde_with_logprob import sde_step_with_logprob
from explorenet.diffusers_patch.sd3_text_encoding import encode_prompt
from explorenet.ema import EMAModuleWrapper
from explorenet.noise_policy import SD3TransformerStdNoisePolicy

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.")

logger = get_logger(__name__)


def _safe_info_value(value, default=0.0):
    if isinstance(value, torch.Tensor):
        return torch.nan_to_num(value.detach().float(), nan=default, posinf=default, neginf=default)
    return torch.tensor(float(value), dtype=torch.float32)


def log_metrics_local(save_dir, metrics_dict, global_step):
    """Append metrics as a JSON line to {save_dir}/metrics.jsonl."""
    metrics_file = os.path.join(save_dir, "metrics.jsonl")
    os.makedirs(save_dir, exist_ok=True)
    entry = {"step": global_step, **metrics_dict}
    with open(metrics_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _as_1d_reward_array(values):
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim > 1:
        arr = arr.reshape(arr.shape[0], -1).mean(axis=1)
    return arr.reshape(-1)


def _threshold_key(value):
    text = f"{float(value):.6g}"
    return text.replace("-", "m").replace(".", "p").replace("+", "").replace("e", "e")


def compute_prompt_group_reward_diagnostics(
    prompts,
    rewards,
    prefix="group_reward_",
    success_threshold=0.5,
    failure_threshold=0.1,
    clear_range_threshold=0.25,
    near_zero_thresholds=(1e-8, 1e-4, 1e-3, 1e-2, 5e-2),
):
    """Logging-only prompt-group reward contrast diagnostics."""
    if rewards is None:
        return {}
    reward_arr = _as_1d_reward_array(rewards)
    if len(prompts) != reward_arr.shape[0] or reward_arr.shape[0] == 0:
        return {}

    grouped = defaultdict(list)
    for idx, prompt in enumerate(prompts):
        grouped[prompt].append(float(reward_arr[idx]))
    if not grouped:
        return {}

    sizes = np.asarray([len(vals) for vals in grouped.values()], dtype=np.float32)
    stds = np.asarray([np.std(vals) for vals in grouped.values()], dtype=np.float32)
    bests = np.asarray([np.max(vals) for vals in grouped.values()], dtype=np.float32)
    worsts = np.asarray([np.min(vals) for vals in grouped.values()], dtype=np.float32)
    means = np.asarray([np.mean(vals) for vals in grouped.values()], dtype=np.float32)
    ranges = bests - worsts
    mixed_success_failure = (bests >= float(success_threshold)) & (worsts <= float(failure_threshold))
    clear_range = ranges >= float(clear_range_threshold)

    out = {
        f"{prefix}num_groups": float(len(grouped)),
        f"{prefix}group_size_mean": float(sizes.mean()),
        f"{prefix}group_size_min": float(sizes.min()),
        f"{prefix}group_size_max": float(sizes.max()),
        f"{prefix}mean_mean": float(means.mean()),
        f"{prefix}std_mean": float(stds.mean()),
        f"{prefix}std_median": float(np.median(stds)),
        f"{prefix}std_max": float(stds.max()),
        f"{prefix}zero_std_ratio": float(np.mean(stds == 0.0)),
        f"{prefix}best_mean": float(bests.mean()),
        f"{prefix}worst_mean": float(worsts.mean()),
        f"{prefix}range_mean": float(ranges.mean()),
        f"{prefix}range_median": float(np.median(ranges)),
        f"{prefix}range_max": float(ranges.max()),
        f"{prefix}has_success_ratio": float(np.mean(bests >= float(success_threshold))),
        f"{prefix}has_failure_ratio": float(np.mean(worsts <= float(failure_threshold))),
        f"{prefix}mixed_success_failure_ratio": float(mixed_success_failure.mean()),
        f"{prefix}clear_range_ratio": float(clear_range.mean()),
    }
    for threshold in near_zero_thresholds:
        out[f"{prefix}near_zero_std_ratio_le_{_threshold_key(threshold)}"] = float(
            np.mean(stds <= float(threshold))
        )
    return out


def summarize_sample_noise_logs(samples):
    """Summarize sample-time noise-policy stats, including timestep structure."""
    out = {}
    timestep_keys = {
        "sample_noise_eps_net_std",
        "sample_noise_eps_mix_std",
        "sample_noise_policy_log_std_mean",
        "sample_noise_policy_log_std_std",
        "sample_noise_policy_log_std_clamp_min_frac",
        "sample_noise_policy_log_std_clamp_max_frac",
        "sample_noise_policy_log_std_channel_mean_std",
        "sample_noise_policy_log_std_spatial_mean_std",
    }
    for key, value in samples.items():
        if not key.startswith("sample_noise_") or not torch.is_tensor(value):
            continue
        tensor = value.detach().float()
        out[key] = float(torch.nan_to_num(tensor.mean(), nan=0.0, posinf=0.0, neginf=0.0).item())
        if key not in timestep_keys or tensor.ndim < 2:
            continue
        per_t = tensor
        while per_t.ndim > 2:
            per_t = per_t.mean(dim=-1)
        per_t = torch.nan_to_num(per_t, nan=0.0, posinf=0.0, neginf=0.0).mean(dim=0)
        if per_t.numel() == 0:
            continue
        mid = per_t.numel() // 2
        early_end = max(1, per_t.numel() // 3)
        late_start = min(per_t.numel() - 1, (2 * per_t.numel()) // 3)
        out[f"{key}_t_first"] = float(per_t[0].item())
        out[f"{key}_t_mid"] = float(per_t[mid].item())
        out[f"{key}_t_last"] = float(per_t[-1].item())
        out[f"{key}_t_early_mean"] = float(per_t[:early_end].mean().item())
        out[f"{key}_t_late_mean"] = float(per_t[late_start:].mean().item())
        out[f"{key}_t_min"] = float(per_t.min().item())
        out[f"{key}_t_max"] = float(per_t.max().item())
    return out


def compute_group_noise_rewards(prompts, rewards):
    """Per-sample noise-policy reward: the within-prompt-group reward std.

    Every sample of a prompt group receives the std of that group's rewards
    across the iteration. The caller standardizes these values over all
    samples to obtain the noise-policy advantage B_p.
    """
    grouped = defaultdict(list)
    for idx, prompt in enumerate(prompts):
        grouped[prompt].append(idx)
    out = np.zeros_like(rewards, dtype=np.float32)
    for idxs in grouped.values():
        vals = rewards[idxs].astype(np.float32)
        out[idxs] = float(np.std(vals))
    return out


_PROMPT_GROUP_KEY_PREFIX = "prompt_sha256_63:"


def _prompt_group_key_int(prompt):
    digest = hashlib.sha256(str(prompt).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _prompt_group_key(prompt):
    prompt = str(prompt)
    if prompt.startswith(_PROMPT_GROUP_KEY_PREFIX):
        return prompt
    return f"{_PROMPT_GROUP_KEY_PREFIX}{_prompt_group_key_int(prompt)}"


def _gather_object_list(local_items, accelerator):
    """Gather Python objects in rank-major order to align with tensor gather."""
    local_items = list(local_items)
    if not dist.is_available() or not dist.is_initialized():
        return local_items
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local_items)
    return [item for rank_items in gathered for item in rank_items]


def _gather_prompt_group_keys(local_prompts, accelerator):
    """Gather stable prompt identity keys without Python object collectives."""
    local_keys = [_prompt_group_key_int(prompt) for prompt in local_prompts]
    if not dist.is_available() or not dist.is_initialized():
        return [f"{_PROMPT_GROUP_KEY_PREFIX}{int(key)}" for key in local_keys]
    local_tensor = torch.as_tensor(local_keys, device=accelerator.device, dtype=torch.long)
    gathered = accelerator.gather(local_tensor).detach().cpu().tolist()
    return [f"{_PROMPT_GROUP_KEY_PREFIX}{int(key)}" for key in gathered]


def _reward_tensor_for_gather(value, *, key, local_count, device):
    if torch.is_tensor(value):
        tensor = value.detach().to(device)
    else:
        tensor = torch.as_tensor(value, device=device)
    if tensor.ndim == 0:
        tensor = tensor.reshape(1).expand(local_count)
    if tensor.shape[0] != local_count:
        raise ValueError(
            f"reward gather: key '{key}' has local first dimension {tensor.shape[0]}, "
            f"expected {local_count}"
        )
    if not torch.is_floating_point(tensor):
        tensor = tensor.float()
    return tensor.contiguous()


def _gather_reward_dict_consistent(reward_by_key, accelerator):
    """Gather reward metrics with a rank-consistent key/order/shape contract."""
    if "ori_avg" in reward_by_key:
        reference = reward_by_key["ori_avg"]
    elif "avg" in reward_by_key:
        reference = reward_by_key["avg"]
    else:
        raise KeyError(f"reward gather: no avg/ori_avg key in {list(reward_by_key.keys())}")
    if torch.is_tensor(reference):
        local_count = int(reference.shape[0]) if reference.ndim else 1
    else:
        reference_arr = np.asarray(reference)
        local_count = int(reference_arr.shape[0]) if reference_arr.ndim else 1

    local_tensors = {
        str(key): _reward_tensor_for_gather(
            value, key=str(key), local_count=local_count, device=accelerator.device
        )
        for key, value in reward_by_key.items()
    }
    local_schema = {
        key: tuple(int(dim) for dim in tensor.shape[1:])
        for key, tensor in local_tensors.items()
    }
    local_payload = {"count": local_count, "schema": local_schema}

    if dist.is_available() and dist.is_initialized():
        payloads = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(payloads, local_payload)
    else:
        payloads = [local_payload]

    counts = {int(payload["count"]) for payload in payloads}
    if len(counts) != 1:
        raise RuntimeError(f"reward gather: inconsistent per-rank local counts: {sorted(counts)}")

    key_shapes = {}
    for payload in payloads:
        for key, suffix in payload["schema"].items():
            suffix = tuple(int(dim) for dim in suffix)
            previous = key_shapes.setdefault(key, suffix)
            if previous != suffix:
                raise RuntimeError(
                    f"reward gather: key '{key}' has inconsistent trailing shapes "
                    f"{previous} vs {suffix}"
                )

    priority = ["avg", "ori_avg"]
    ordered_keys = [key for key in priority if key in key_shapes]
    ordered_keys.extend(sorted(key for key in key_shapes if key not in set(ordered_keys)))

    gathered = {}
    for key in ordered_keys:
        suffix = key_shapes[key]
        tensor = local_tensors.get(key)
        if tensor is None:
            tensor = torch.full(
                (local_count, *suffix),
                -10.0,
                device=accelerator.device,
                dtype=torch.float32,
            )
        gathered[key] = accelerator.gather(tensor).detach().cpu().numpy()
    return gathered


def _reward_value_to_float(value):
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.ndim != 0:
            return None
        value = value.item()
    if isinstance(value, np.ndarray):
        if value.shape != ():
            return None
        value = value.item()
    if isinstance(value, (np.floating, np.integer, float, int)):
        value = float(value)
        if np.isfinite(value) and value != -10:
            return value
    return None


def _reward_values_to_numpy(value, *, key, expected_len, context):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    elif isinstance(value, (list, tuple)):
        converted = []
        for item in value:
            if torch.is_tensor(item):
                item = item.detach().cpu()
                converted.append(item.item() if item.ndim == 0 else item.numpy())
            else:
                converted.append(item)
        value = converted
    try:
        arr = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{context}: reward key '{key}' must be numeric, got {type(value).__name__}"
        ) from exc
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError(
            f"{context}: reward key '{key}' must be a 1D numeric vector, got shape {arr.shape}"
        )
    if expected_len is not None and len(arr) != expected_len:
        raise ValueError(
            f"{context}: reward key '{key}' has length {len(arr)}, expected {expected_len}"
        )
    if not np.all(np.isfinite(arr[arr != -10])):
        raise ValueError(f"{context}: reward key '{key}' contains non-finite values")
    return arr


def _gather_1d_numpy_variable_length(arr, accelerator, device, *, context):
    """Gather rank-local 1D metric vectors without assuming equal local lengths."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"{context}: expected a 1D metric vector, got shape {arr.shape}")
    if not dist.is_available() or not dist.is_initialized():
        return arr

    world_size = dist.get_world_size()
    local_len = torch.tensor([len(arr)], device=device, dtype=torch.long)
    length_tensors = [torch.zeros_like(local_len) for _ in range(world_size)]
    dist.all_gather(length_tensors, local_len)
    lengths = torch.cat(length_tensors).detach().cpu().numpy().astype(np.int64)
    max_len = int(lengths.max()) if len(lengths) else len(arr)
    if max_len == 0:
        return np.asarray([], dtype=np.float32)

    padded = torch.full((max_len,), -10.0, device=device, dtype=torch.float32)
    if len(arr):
        padded[: len(arr)] = torch.as_tensor(arr, device=device, dtype=torch.float32)
    gathered_tensors = [torch.empty_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, padded)
    gathered = torch.stack(gathered_tensors, dim=0).detach().cpu().numpy()
    return np.concatenate(
        [gathered[rank_idx, :length] for rank_idx, length in enumerate(lengths) if length > 0]
    ).astype(np.float32, copy=False)


def _reward_caption_items(reward_by_key):
    items = []
    for key, value in reward_by_key.items():
        value = _reward_value_to_float(value)
        if value is None:
            continue
        items.append(f"{key}: {value:.2f}")
    return items


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds


def calculate_zero_std_ratio(prompts, gathered_rewards):
    """Fraction of prompt groups whose reward std is zero, and the mean group std."""
    prompt_array = np.array(prompts)
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array,
        return_inverse=True,
        return_counts=True
    )
    grouped_rewards = gathered_rewards['ori_avg'][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    return zero_std_ratio, prompt_std_devs.mean()


def compute_log_prob(transformer, pipeline, sample, j, embeds, pooled_embeds, config):
    # CFG: the negative and positive halves go through one batched forward pass.
    noise_pred = transformer(
        hidden_states=torch.cat([sample["latents"][:, j]] * 2),
        timestep=torch.cat([sample["timesteps"][:, j]] * 2),
        encoder_hidden_states=embeds,
        pooled_projections=pooled_embeds,
        return_dict=False,
    )[0]
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = (
        noise_pred_uncond
        + config.sample.guidance_scale
        * (noise_pred_text - noise_pred_uncond)
    )

    # compute the log prob of next_latents given latents under the current model
    prev_sample, log_prob, prev_sample_mean, std_dev_t = sde_step_with_logprob(
        pipeline.scheduler,
        noise_pred.float(),
        sample["timesteps"][:, j],
        sample["latents"][:, j].float(),
        prev_sample=sample["next_latents"][:, j].float(),
        noise_level=config.sample.noise_level,
    )

    return prev_sample, log_prob, prev_sample_mean, std_dev_t


def eval(
    pipeline,
    test_dataloader,
    text_encoders,
    tokenizers,
    config,
    accelerator,
    global_step,
    reward_fn,
    executor,
    autocast,
    ema,
    transformer_trainable_parameters,
    fixed_eval_prompts=None,
    fixed_eval_metadatas=None,
):
    """Held-out GenEval2 evaluation with deterministic (ODE) sampling and EMA weights."""
    device = accelerator.device
    is_main_process = bool(accelerator.is_main_process)
    is_local_main_process = bool(accelerator.is_local_main_process)
    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers, max_sequence_length=128, device=device)

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.test_batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.test_batch_size, 1)

    all_rewards = defaultdict(list)
    num_eval_batches = 0
    for batch_idx, test_batch in enumerate(tqdm(
            test_dataloader,
            desc="Eval: ",
            disable=not is_local_main_process,
            position=0,
        )):
        prompts, prompt_metadata = test_batch
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts,
            text_encoders,
            tokenizers,
            max_sequence_length=128,
            device=device
        )
        # The last batch may not be full batch_size
        neg_embeds = sample_neg_prompt_embeds[:len(prompt_embeds)]
        neg_pooled = sample_neg_pooled_prompt_embeds[:len(prompt_embeds)]
        with autocast():
            with torch.no_grad():
                images, _, _ = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=neg_embeds,
                    negative_pooled_prompt_embeds=neg_pooled,
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution,
                    noise_level=0,
                )
        rewards = executor.submit(reward_fn, images, prompts, prompt_metadata)
        # yield to to make sure reward computation starts
        time.sleep(0)
        rewards, reward_metadata = rewards.result()

        rewards_np = {
            key: _reward_values_to_numpy(
                value,
                key=key,
                expected_len=len(prompts),
                context=f"eval batch {batch_idx}",
            )
            for key, value in rewards.items()
        }
        for key, arr in rewards_np.items():
            all_rewards[key].append(arr)
        num_eval_batches += 1

    if is_main_process:
        logger.info(f"Eval loop complete: {num_eval_batches} batches")
    local_primary_rewards = {
        key: np.concatenate(value).astype(np.float32, copy=False)
        for key, value in all_rewards.items()
    }
    primary_reward_keys = sorted(set(_gather_object_list(local_primary_rewards.keys(), accelerator)))
    all_rewards = {
        key: _gather_1d_numpy_variable_length(
            local_primary_rewards.get(key, np.asarray([], dtype=np.float32)),
            accelerator,
            device,
            context=f"eval reward key {key}",
        )
        for key in primary_reward_keys
    }

    eval_metric_log = {}
    for key, value in all_rewards.items():
        valid_mask = np.isfinite(value) & (value != -10)
        mean_value = float(np.mean(value[valid_mask])) if np.any(valid_mask) else float("nan")
        eval_metric_log[f"eval/{key}"] = mean_value
        eval_metric_log[f"eval/{key}_count"] = int(np.count_nonzero(valid_mask))

    # Fixed-prompt visualization on the main process (W&B images only).
    if is_main_process and fixed_eval_prompts:
        num_vis = len(fixed_eval_prompts)
        vis_prompts = fixed_eval_prompts
        vis_metadatas = fixed_eval_metadatas
        vis_neg_prompt_embeds = neg_prompt_embed.repeat(num_vis, 1, 1)
        vis_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(num_vis, 1)
        vis_prompt_embeds, vis_pooled_prompt_embeds = compute_text_embeddings(
            vis_prompts, text_encoders, tokenizers,
            max_sequence_length=128, device=device
        )
        with autocast():
            with torch.no_grad():
                vis_images, _, _ = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=vis_prompt_embeds,
                    pooled_prompt_embeds=vis_pooled_prompt_embeds,
                    negative_prompt_embeds=vis_neg_prompt_embeds,
                    negative_pooled_prompt_embeds=vis_neg_pooled_prompt_embeds,
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution,
                    noise_level=0,
                )
        vis_rewards = executor.submit(reward_fn, vis_images, vis_prompts, vis_metadatas)
        vis_rewards, _ = vis_rewards.result()
        vis_images_np = vis_images.cpu().numpy()

        with tempfile.TemporaryDirectory() as tmpdir:
            for idx in range(num_vis):
                pil = Image.fromarray(
                    (vis_images_np[idx].transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil = pil.resize((config.resolution, config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))
            sampled_rewards = [{k: vis_rewards[k][idx] for k in vis_rewards} for idx in range(num_vis)]
            wandb.log(
                {
                    "eval_images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompt:.1000} | " + " | ".join(_reward_caption_items(reward)),
                        )
                        for idx, (prompt, reward) in enumerate(zip(vis_prompts, sampled_rewards))
                    ],
                    **eval_metric_log,
                },
                step=global_step,
            )
    elif is_main_process:
        wandb.log(eval_metric_log, step=global_step)
    if is_main_process:
        log_metrics_local(config.save_dir, {"type": "eval", **eval_metric_log}, global_step)
    torch.cuda.empty_cache()
    accelerator.wait_for_everyone()
    if config.train.ema:
        ema.copy_temp_to(transformer_trainable_parameters)


def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def _capture_rng_state(accelerator):
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (
            numpy_state[0],
            numpy_state[1].tolist(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ),
        "torch_cpu": torch.get_rng_state().cpu(),
        "torch_cuda": (
            torch.cuda.get_rng_state(accelerator.device).cpu()
            if torch.cuda.is_available()
            else None
        ),
    }


def _restore_rng_state(rng_state, accelerator):
    random.setstate(rng_state["python"])
    numpy_state = rng_state["numpy"]
    np.random.set_state(
        (
            numpy_state[0],
            np.asarray(numpy_state[1], dtype=np.uint32),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        )
    )
    torch.set_rng_state(rng_state["torch_cpu"])
    if torch.cuda.is_available() and rng_state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state(rng_state["torch_cuda"], device=accelerator.device)


def save_ckpt(save_dir, transformer, global_step, accelerator, ema,
              transformer_trainable_parameters, config, optimizer=None,
              epoch=None, stat_tracker=None,
              noise_policy_module=None, noise_optimizer=None):
    """Save the (EMA) denoiser LoRA, optimizer and training state, and the noise policy."""
    save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
    save_root_lora = os.path.join(save_root, "lora")
    if accelerator.is_main_process:
        os.makedirs(save_root_lora, exist_ok=True)
    accelerator.wait_for_everyone()

    local_rng_state = _capture_rng_state(accelerator)
    if dist.is_available() and dist.is_initialized():
        rng_states_by_rank = (
            [None] * accelerator.num_processes
            if accelerator.is_main_process
            else None
        )
        dist.gather_object(local_rng_state, rng_states_by_rank, dst=0)
    else:
        rng_states_by_rank = [local_rng_state]

    if accelerator.is_main_process:
        if config.train.ema:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
        unwrap_model(transformer, accelerator).save_pretrained(save_root_lora)
        if config.train.ema:
            ema.copy_temp_to(transformer_trainable_parameters)
        training_state = {
            "global_step": global_step,
            "epoch": epoch,
            "wandb_run_id": wandb.run.id if wandb.run else None,
            "rng_world_size": accelerator.num_processes,
            "rng_states_by_rank": rng_states_by_rank,
        }
        if config.train.ema:
            training_state["ema_state_dict"] = ema.state_dict()
        if stat_tracker is not None:
            training_state["stat_tracker"] = {
                "stats": {k: v.tolist() if hasattr(v, 'tolist') else list(v)
                          for k, v in stat_tracker.stats.items()},
                "history_prompts": list(stat_tracker.history_prompts),
            }
        torch.save(training_state, os.path.join(save_root, "training_state.pt"))
        if optimizer is not None:
            torch.save(optimizer.state_dict(), os.path.join(save_root, "optimizer.pt"))
        if noise_policy_module is not None:
            np_module = accelerator.unwrap_model(noise_policy_module)
            torch.save(np_module.state_dict(), os.path.join(save_root, "noise_policy_module.pt"))
        if noise_optimizer is not None:
            torch.save(noise_optimizer.state_dict(), os.path.join(save_root, "noise_optimizer.pt"))
        print(f"Saved full checkpoint to {save_root} (epoch={epoch}, step={global_step})")


def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    # A fresh run (no lora_path) must not write into a directory that already
    # holds a run: checkpoints would be overwritten and metrics.jsonl appended to.
    if (
        config.save_dir
        and not config.train.lora_path
        and os.environ.get("EXPLORENET_ALLOW_EXISTING_SAVE_DIR") != "1"
        and (
            os.path.exists(os.path.join(config.save_dir, "metrics.jsonl"))
            or os.path.isdir(os.path.join(config.save_dir, "checkpoints"))
        )
    ):
        raise RuntimeError(
            f"Refusing to start a fresh run in existing run directory {config.save_dir}. "
            "Choose a new save_dir, or set EXPLORENET_ALLOW_EXISTING_SAVE_DIR=1 if this is intentional."
        )

    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    process_group_kwargs = InitProcessGroupKwargs(
        timeout=datetime.timedelta(minutes=120),
    )
    ddp_kwargs = DistributedDataParallelKwargs(
        broadcast_buffers=False,
        find_unused_parameters=False,
    )

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
        kwargs_handlers=[process_group_kwargs, ddp_kwargs],
    )
    if config.num_gpus and accelerator.num_processes != int(config.num_gpus):
        raise RuntimeError(
            f"Config batch schedule was built for num_gpus={config.num_gpus}, but "
            f"{accelerator.num_processes} processes were launched. Select the config with "
            f"`--config=config/explorenet_sd35.py:<name>,num_gpus={accelerator.num_processes}`."
        )
    if accelerator.is_main_process:
        # Continue the W&B run of the checkpoint when resuming.
        _wandb_resume_id = None
        if config.train.lora_path:
            _ckpt_root = os.path.dirname(config.train.lora_path.rstrip("/"))
            _ts_path = os.path.join(_ckpt_root, "training_state.pt")
            if os.path.exists(_ts_path):
                _ts = torch.load(_ts_path, map_location="cpu")
                _wandb_resume_id = _ts.get("wandb_run_id")
        _wandb_settings = wandb.Settings(init_timeout=int(os.environ.get("WANDB_INIT_TIMEOUT", "900")))
        _wandb_project = os.environ.get("WANDB_PROJECT", "explorenet")
        # Honor WANDB_MODE=online/offline/disabled explicitly (an explicit Settings object would otherwise take precedence).
        _wandb_mode = os.environ.get("WANDB_MODE")
        _wandb_mode_kwargs = {"mode": _wandb_mode} if _wandb_mode in {"online", "offline", "disabled"} else {}
        if _wandb_resume_id:
            wandb.init(
                project=_wandb_project,
                id=_wandb_resume_id,
                resume="allow",
                settings=_wandb_settings,
                **_wandb_mode_kwargs,
            )
        else:
            wandb.init(
                project=_wandb_project,
                name=config.run_name,
                config=config.to_dict(),
                settings=_wandb_settings,
                **_wandb_mode_kwargs,
            )
    accelerator.wait_for_everyone()
    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    # load scheduler, tokenizer and models.
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        config.pretrained.model
    )
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    pipeline.transformer.requires_grad_(False)

    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]

    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=torch.float32)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_2.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_3.to(accelerator.device, dtype=inference_dtype)

    pipeline.transformer.to(accelerator.device)

    sd3_lora_target_modules = [
        "attn.add_k_proj",
        "attn.add_q_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "attn.to_k",
        "attn.to_out.0",
        "attn.to_q",
        "attn.to_v",
    ]
    transformer_lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        init_lora_weights="gaussian",
        target_modules=sd3_lora_target_modules,
    )
    if config.train.lora_path:
        pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, config.train.lora_path)
        # After loading with PeftModel.from_pretrained, all parameters have requires_grad set to False. You need to call set_adapter to enable gradients for the adapter parameters.
        pipeline.transformer.set_adapter("default")
    else:
        pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)

    transformer = pipeline.transformer
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    # --- ExploreNet: learned SDE noise policy g_phi ---
    # A second SD3.5 transformer initialized from the pretrained weights, with
    # its own LoRA, kept in fp32. It is only used during training.
    noise_policy_module = None
    noise_policy_parameters = []
    if config.sample.use_learned_noise_policy:
        noise_transformer = SD3Transformer2DModel.from_pretrained(
            config.pretrained.model,
            subfolder="transformer",
        )
        noise_transformer.requires_grad_(False)
        noise_lora_config = LoraConfig(
            r=config.sample.noise_policy_lora_rank,
            lora_alpha=config.sample.noise_policy_lora_alpha,
            init_lora_weights="gaussian",
            target_modules=sd3_lora_target_modules,
        )
        noise_transformer = get_peft_model(noise_transformer, noise_lora_config)
        noise_policy_module = SD3TransformerStdNoisePolicy(
            noise_transformer,
            log_std_min=config.sample.noise_policy_log_std_min,
            log_std_max=config.sample.noise_policy_log_std_max,
        ).to(accelerator.device)
        noise_policy_module.transformer.to(accelerator.device, dtype=torch.float32)
        if config.train.lora_path:
            ckpt_root = os.path.dirname(config.train.lora_path.rstrip("/"))
            noise_ckpt_path = os.path.join(ckpt_root, "noise_policy_module.pt")
            if os.path.exists(noise_ckpt_path):
                noise_policy_module.load_state_dict(
                    torch.load(noise_ckpt_path, map_location="cpu")
                )
                if accelerator.is_main_process:
                    print(f"Loaded noise_policy_module from {noise_ckpt_path}")
        noise_policy_parameters = list(filter(lambda p: p.requires_grad, noise_policy_module.parameters()))

    # This ema setting affects the previous 20 x 8 = 160 steps on average.
    ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=8, device=accelerator.device)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )
    noise_optimizer = None
    if noise_policy_module is not None and noise_policy_parameters:
        noise_optimizer = torch.optim.AdamW(
            noise_policy_parameters,
            lr=config.train.noise_policy_learning_rate,
            betas=(config.train.adam_beta1, config.train.adam_beta2),
            weight_decay=config.train.noise_policy_weight_decay,
            eps=config.train.adam_epsilon,
        )

    # prepare prompt and reward fn
    reward_fn = explorenet.rewards.multi_score(accelerator.device, config.reward_fn)
    eval_reward_fn = explorenet.rewards.multi_score(accelerator.device, config.reward_fn)

    train_dataset = GenevalPromptDataset(config.dataset, 'train')
    test_dataset = GenevalPromptDataset(config.dataset, 'test')

    train_sampler = DistributedKRepeatSampler(
        dataset=train_dataset,
        batch_size=config.sample.train_batch_size,
        k=config.sample.num_image_per_prompt,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        seed=42,
        rollout_batches=config.sample.num_batches_per_epoch,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        # The sampler state is updated before each rollout batch. A worker
        # process can prefetch with stale sampler state and repeat prompts.
        num_workers=0,
        collate_fn=GenevalPromptDataset.collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.sample.test_batch_size,
        collate_fn=GenevalPromptDataset.collate_fn,
        shuffle=False,
        num_workers=8,
    )

    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device)

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.train_batch_size, 1)
    train_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.train.batch_size, 1)

    # initialize stat tracker
    stat_tracker = PerPromptStatTracker(config.sample.global_std)

    # autocast is not needed for LoRA training and uses more memory
    autocast = contextlib.nullcontext

    # Prepare everything with our `accelerator`.
    if noise_policy_module is not None:
        transformer, optimizer, noise_policy_module, noise_optimizer, test_dataloader = accelerator.prepare(
            transformer, optimizer, noise_policy_module, noise_optimizer, test_dataloader
        )
    else:
        transformer, optimizer, test_dataloader = accelerator.prepare(transformer, optimizer, test_dataloader)

    # Executor for asynchronous reward callbacks.
    reward_executor_workers = int(os.environ.get("REWARD_EXECUTOR_MAX_WORKERS", "8"))
    executor = futures.ThreadPoolExecutor(max_workers=reward_executor_workers)
    accelerator.print(f"Reward executor max workers = {reward_executor_workers}")

    # Train!
    samples_per_epoch = (
        config.sample.train_batch_size
        * accelerator.num_processes
        * config.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train.batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")

    epoch = 0
    global_step = 0
    train_iter = iter(train_dataloader)

    # --- Resume from checkpoint if training_state.pt exists ---
    if config.train.lora_path:
        # lora_path points to the lora/ subdir; training_state is one level up
        ckpt_root = os.path.dirname(config.train.lora_path.rstrip("/"))
        training_state_path = os.path.join(ckpt_root, "training_state.pt")
        optimizer_state_path = os.path.join(ckpt_root, "optimizer.pt")
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, map_location="cpu")
            epoch = training_state["epoch"]
            global_step = training_state["global_step"]
            if config.train.ema and "ema_state_dict" in training_state:
                ema.load_state_dict(training_state["ema_state_dict"])
            if "stat_tracker" in training_state:
                st_data = training_state["stat_tracker"]
                stat_tracker.stats = {
                    _prompt_group_key(k): np.array(v) for k, v in st_data["stats"].items()
                }
                stat_tracker.history_prompts = set(st_data.get("history_prompts", []))
                stat_tracker.history_prompts.update(hash(k) for k in stat_tracker.stats.keys())
            if os.path.exists(optimizer_state_path):
                optimizer.load_state_dict(
                    torch.load(optimizer_state_path, map_location="cpu")
                )
            if noise_optimizer is not None:
                noise_optimizer_state_path = os.path.join(ckpt_root, "noise_optimizer.pt")
                if os.path.exists(noise_optimizer_state_path):
                    noise_optimizer.load_state_dict(
                        torch.load(noise_optimizer_state_path, map_location="cpu")
                    )
            # Fast-forward the data sampler to the correct position.
            for e in range(epoch):
                for i in range(config.sample.num_batches_per_epoch):
                    train_sampler.set_epoch(e * config.sample.num_batches_per_epoch + i)
                    next(train_iter)
            rng_states_by_rank = training_state.get("rng_states_by_rank")
            if rng_states_by_rank is None:
                if accelerator.is_main_process:
                    print("Checkpoint has no per-rank RNG state; continuing with the current RNG state.")
            else:
                saved_world_size = int(
                    training_state.get("rng_world_size", len(rng_states_by_rank))
                )
                if saved_world_size != accelerator.num_processes:
                    raise RuntimeError(
                        "Cannot exactly resume with a different world size: "
                        f"checkpoint={saved_world_size}, "
                        f"current={accelerator.num_processes}"
                    )
                _restore_rng_state(
                    rng_states_by_rank[accelerator.process_index], accelerator
                )
            if accelerator.is_main_process:
                print(f"Resumed from checkpoint: epoch={epoch}, global_step={global_step}")
                wandb.log({"resumed_from_epoch": epoch, "resumed_from_step": global_step}, step=global_step)

    # Fixed held-out prompts for W&B image visualization at each evaluation.
    fixed_eval_metadatas = []
    fixed_eval_prompts = []
    fixed_eval_count = int(config.sample.test_batch_size)
    with open(os.path.join(config.dataset, "test_metadata.jsonl"), "r", encoding="utf-8") as handle:
        for line in handle:
            if len(fixed_eval_prompts) >= fixed_eval_count:
                break
            metadata = json.loads(line)
            fixed_eval_metadatas.append(metadata)
            fixed_eval_prompts.append(metadata["prompt"])

    while True:
        #################### EVAL ####################
        pipeline.transformer.eval()
        if epoch % config.eval_freq == 0:
            eval(
                pipeline,
                test_dataloader,
                text_encoders,
                tokenizers,
                config,
                accelerator,
                global_step,
                eval_reward_fn,
                executor,
                autocast,
                ema,
                transformer_trainable_parameters,
                fixed_eval_prompts=fixed_eval_prompts,
                fixed_eval_metadatas=fixed_eval_metadatas,
            )
        if epoch % config.save_freq == 0 and epoch > 0:
            save_ckpt(config.save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config,
                      optimizer=optimizer, epoch=epoch,
                      stat_tracker=stat_tracker,
                      noise_policy_module=noise_policy_module,
                      noise_optimizer=noise_optimizer)
        accelerator.wait_for_everyone()
        max_global_step = config.max_global_step
        if max_global_step is not None and global_step >= int(max_global_step):
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                print(
                    f"Reached max_global_step={int(max_global_step)} at "
                    f"epoch={epoch}, global_step={global_step}; stopping.",
                    flush=True,
                )
                wandb.finish()
            return

        #################### SAMPLING ####################
        pipeline.transformer.eval()
        if noise_policy_module is not None:
            noise_policy_module.eval()
        samples = []
        prompts = []
        local_sample_prompts = []
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            prompts, prompt_metadata = next(train_iter)
            local_sample_prompts.extend(prompts)

            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                prompts,
                text_encoders,
                tokenizers,
                max_sequence_length=128,
                device=accelerator.device
            )

            # sample
            with autocast():
                with torch.no_grad():
                    pipeline_out = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds,
                        negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution,
                        noise_level=config.sample.noise_level,
                        generator=None,
                        max_sequence_length=128,
                        learned_noise_module=accelerator.unwrap_model(noise_policy_module) if noise_policy_module is not None else None,
                        learned_noise_alpha=config.sample.noise_policy_alpha if noise_policy_module is not None else 0.0,
                    )
                    if noise_policy_module is not None:
                        images, latents, log_probs, learned_noise_info = pipeline_out
                    else:
                        images, latents, log_probs = pipeline_out
                        learned_noise_info = None

            latents = torch.stack(
                latents, dim=1
            )  # (batch_size, num_steps + 1, 16, 64, 64)
            log_probs = torch.stack(log_probs, dim=1)  # shape after stack (batch_size, num_steps)

            timesteps = pipeline.scheduler.timesteps.repeat(
                config.sample.train_batch_size, 1
            )  # (batch_size, num_steps)

            # compute rewards asynchronously
            rewards = executor.submit(reward_fn, images, prompts, prompt_metadata)
            # yield to to make sure reward computation starts
            time.sleep(0)

            noise_fields = {}
            if learned_noise_info is not None:
                noise_fields["noise_actions"] = torch.stack(learned_noise_info["noise_actions"], dim=1)
                noise_fields["noise_action_log_probs"] = torch.stack(learned_noise_info["noise_log_probs"], dim=1)
                for k in learned_noise_info["noise_stats"][0]:
                    noise_fields[f"sample_{k}"] = torch.stack(
                        [step_stats[k] for step_stats in learned_noise_info["noise_stats"]], dim=1
                    )

            samples.append(
                {
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                    "timesteps": timesteps,
                    "latents": latents[
                        :, :-1
                    ],  # each entry is the latent before timestep t
                    "next_latents": latents[
                        :, 1:
                    ],  # each entry is the latent after timestep t
                    "log_probs": log_probs,
                    "rewards": rewards,
                    **noise_fields,
                }
            )

        # wait for all rewards to be computed
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards, reward_metadata = sample["rewards"].result()
            expected_len = int(sample["prompt_embeds"].shape[0])
            sample["rewards"] = {
                key: torch.as_tensor(
                    _reward_values_to_numpy(
                        value,
                        key=key,
                        expected_len=expected_len,
                        context="training reward",
                    ),
                    device=accelerator.device,
                ).float()
                for key, value in rewards.items()
            }

        # Collate samples into dict where each entry has shape
        # (num_batches_per_epoch * sample.batch_size, ...).
        collated_samples = {}
        for k in samples[0].keys():
            if not isinstance(samples[0][k], dict):
                collated_samples[k] = torch.cat([s[k] for s in samples], dim=0)
                continue
            sub_keys = sorted({sub_key for s in samples for sub_key in s[k].keys()})
            collated_samples[k] = {
                sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0) for sub_key in sub_keys
            }
        samples = collated_samples

        if noise_policy_module is not None and accelerator.is_main_process:
            sample_noise_log = summarize_sample_noise_logs(samples)
            wandb.log(sample_noise_log, step=global_step)
            log_metrics_local(config.save_dir, {"type": "noise_sample", "epoch": epoch, **sample_noise_log}, global_step)

        accelerator.wait_for_everyone()

        # Restore scheduler timesteps after evaluation (which uses eval_num_steps)
        # so the training loop can look up the original sampling timesteps.
        pipeline.scheduler.set_timesteps(config.sample.num_steps, device=accelerator.device)

        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"]
        # Repeat the reward along the timestep dimension (timestep-independent advantage).
        samples["rewards"]["avg"] = samples["rewards"]["avg"].unsqueeze(1).repeat(1, num_train_timesteps)
        # gather rewards across processes
        gathered_rewards = _gather_reward_dict_consistent(samples["rewards"], accelerator)
        gathered_prompts_all = _gather_prompt_group_keys(local_sample_prompts, accelerator)
        expected_prompt_count = next(iter(gathered_rewards.values())).shape[0]
        if len(gathered_prompts_all) != expected_prompt_count:
            raise RuntimeError(
                "Gathered prompt group-key count does not match gathered rewards: "
                f"prompts={len(gathered_prompts_all)} rewards={expected_prompt_count}"
            )
        unique_prompt_count = len(set(gathered_prompts_all))
        expected_unique_prompt_count = expected_prompt_count // max(1, int(config.sample.num_image_per_prompt))
        if accelerator.is_main_process:
            prompt_group_log = {
                "train_unique_prompt_count": int(unique_prompt_count),
                "train_expected_unique_prompt_count": int(expected_unique_prompt_count),
                "train_unique_prompt_ratio": float(unique_prompt_count / max(1, expected_unique_prompt_count)),
            }
            wandb.log(prompt_group_log, step=global_step)
            log_metrics_local(config.save_dir, {"type": "train_prompt_groups", "epoch": epoch, **prompt_group_log}, global_step)
        configured_reward_keys = list(config.reward_fn.keys())
        primary_reward_key = next(
            (key for key in configured_reward_keys if key in gathered_rewards),
            "ori_avg",
        )
        primary_reward_values = gathered_rewards.get(
            primary_reward_key,
            gathered_rewards.get("ori_avg", gathered_rewards.get("avg")),
        )
        reward_key_for_log = str(primary_reward_key).replace("/", "_")
        train_group_diag = compute_prompt_group_reward_diagnostics(
            gathered_prompts_all,
            primary_reward_values,
            prefix=f"group_{reward_key_for_log}_",
        )
        # log rewards
        if accelerator.is_main_process:
            def _valid_train_reward_mean(value):
                arr = np.asarray(value, dtype=np.float32)
                valid = arr[arr != -10]
                return float(np.mean(valid)) if valid.size else float("nan")
            reward_mean_log = {
                f"reward_{key}": _valid_train_reward_mean(value)
                for key, value in gathered_rewards.items()
            }
            wandb.log({"epoch": epoch, **reward_mean_log, **train_group_diag}, step=global_step)
            log_metrics_local(
                config.save_dir,
                {"type": "train_reward", "epoch": epoch, **reward_mean_log, **train_group_diag},
                global_step,
            )

        if noise_policy_module is not None:
            # ExploreNet advantage B_p: the within-group reward std of each
            # prompt group, standardized over all samples of the iteration.
            primary_rewards_for_noise = gathered_rewards.get(
                primary_reward_key,
                gathered_rewards.get("avg", gathered_rewards.get("ori_avg")),
            )
            noise_rewards_np = compute_group_noise_rewards(
                gathered_prompts_all,
                primary_rewards_for_noise.astype(np.float32),
            )
            noise_adv_np = (noise_rewards_np - noise_rewards_np.mean()) / (noise_rewards_np.std() + 1e-4)
            noise_adv = torch.as_tensor(noise_adv_np, device=accelerator.device)
            samples["noise_advantages"] = (
                noise_adv.reshape(accelerator.num_processes, -1)[accelerator.process_index]
                .unsqueeze(1)
                .repeat(1, num_train_timesteps)
            )
            if accelerator.is_main_process:
                noise_reward_log = {
                    "noise_reward_mean": float(noise_rewards_np.mean()),
                    "noise_reward_std": float(noise_rewards_np.std()),
                }
                wandb.log(noise_reward_log, step=global_step)
                log_metrics_local(config.save_dir, {"type": "noise_reward", "epoch": epoch, **noise_reward_log}, global_step)

        # per-prompt mean/std tracking
        prompts = gathered_prompts_all
        advantages = stat_tracker.update(prompts, gathered_rewards['avg'])
        if accelerator.is_local_main_process:
            print("len(prompts)", len(prompts))
            print("len unique prompts", len(set(prompts)))
            avg_rewards = gathered_rewards['avg']
            print(f"reward mean: {avg_rewards.mean():.4f}, std: {avg_rewards.std():.4f}, min: {avg_rewards.min():.4f}, max: {avg_rewards.max():.4f}")

        group_size, trained_prompt_num = stat_tracker.get_stats()

        zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(prompts, gathered_rewards)

        if accelerator.is_local_main_process:
            print(f"zero_std_ratio: {zero_std_ratio:.4f}, reward_std_mean: {reward_std_mean:.6f}")

        if accelerator.is_main_process:
            wandb.log(
                {
                    "group_size": group_size,
                    "trained_prompt_num": trained_prompt_num,
                    "zero_std_ratio": zero_std_ratio,
                    "reward_std_mean": reward_std_mean,
                },
                step=global_step,
            )
        stat_tracker.clear()

        # ungather advantages; we only need to keep the entries corresponding to the samples on this process
        advantages = torch.as_tensor(advantages)
        samples["advantages"] = (
            advantages.reshape(accelerator.num_processes, -1, advantages.shape[-1])[accelerator.process_index]
            .to(accelerator.device)
        )
        if accelerator.is_local_main_process:
            print("advantages: ", samples["advantages"].abs().mean())

        del samples["rewards"]

        # Get the mask for samples where all advantages are zero across the time dimension
        mask = (samples["advantages"].abs().sum(dim=1) != 0)

        # If the number of True values in mask is not divisible by config.sample.num_batches_per_epoch,
        # randomly change some False values to True to make it divisible
        num_batches = config.sample.num_batches_per_epoch
        true_count = mask.sum()
        if true_count == 0:
            # Keep every rank in the same DDP collective sequence even if its
            # local rewards have zero within-group advantage.
            pad_indices = torch.arange(len(mask), device=mask.device)
            mask[pad_indices[:num_batches]] = True
            true_count = mask.sum()
        if true_count % num_batches != 0:
            false_indices = torch.where(~mask)[0]
            num_to_change = num_batches - (true_count % num_batches)
            if len(false_indices) >= num_to_change:
                random_indices = torch.randperm(len(false_indices))[:num_to_change]
                mask[false_indices[random_indices]] = True
        if accelerator.is_main_process:
            wandb.log(
                {
                    "actual_batch_size": mask.sum().item()//config.sample.num_batches_per_epoch,
                },
                step=global_step,
            )
        # Filter out samples where the entire time dimension of advantages is zero
        samples = {k: v[mask] for k, v in samples.items()}

        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert num_timesteps == config.sample.num_steps

        #################### TRAINING ####################
        for inner_epoch in range(config.train.num_inner_epochs):
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            samples = {k: v[perm] for k, v in samples.items()}

            # rebatch for training
            samples_batched = {
                k: v.reshape(-1, total_batch_size//config.sample.num_batches_per_epoch, *v.shape[1:])
                for k, v in samples.items()
            }

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            # train
            pipeline.transformer.train()
            if noise_policy_module is not None:
                noise_policy_module.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                # concat negative prompts to sample prompts to avoid two forward passes
                embeds = torch.cat(
                    [train_neg_prompt_embeds[:len(sample["prompt_embeds"])], sample["prompt_embeds"]]
                )
                pooled_embeds = torch.cat(
                    [train_neg_pooled_prompt_embeds[:len(sample["pooled_prompt_embeds"])], sample["pooled_prompt_embeds"]]
                )

                train_timesteps = [step_index for step_index in range(num_train_timesteps)]
                for j in tqdm(
                    train_timesteps,
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    has_noise_actions = noise_policy_module is not None and "noise_actions" in sample
                    accumulate_models = [transformer]
                    if has_noise_actions:
                        accumulate_models.append(noise_policy_module)
                    with accelerator.accumulate(*accumulate_models):
                        with autocast():
                            prev_sample, log_prob, prev_sample_mean, std_dev_t = compute_log_prob(transformer, pipeline, sample, j, embeds, pooled_embeds, config)
                            if config.train.beta > 0:
                                with torch.no_grad():
                                    with accelerator.unwrap_model(transformer).disable_adapter():
                                        _, _, prev_sample_mean_ref, _ = compute_log_prob(transformer, pipeline, sample, j, embeds, pooled_embeds, config)

                        # grpo logic
                        advantages = torch.clamp(
                            sample["advantages"][:, j],
                            -config.train.adv_clip_max,
                            config.train.adv_clip_max,
                        )
                        log_ratio = log_prob - sample["log_probs"][:, j]
                        log_ratio = torch.nan_to_num(log_ratio.float(), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
                        ratio = torch.exp(log_ratio)
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(
                            ratio,
                            1.0 - config.train.clip_range,
                            1.0 + config.train.clip_range,
                        )
                        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                        if config.train.beta > 0:
                            kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=(1,2,3), keepdim=True) / (2 * std_dev_t ** 2)
                            kl_loss = torch.mean(kl_loss)
                            loss = policy_loss + config.train.beta * kl_loss
                        else:
                            loss = policy_loss

                        info["approx_kl"].append(
                            0.5
                            * torch.mean(log_ratio ** 2)
                        )
                        info["clipfrac"].append(
                            torch.mean(
                                (
                                    torch.abs(ratio - 1.0) > config.train.clip_range
                                ).float()
                            )
                        )
                        info["clipfrac_gt_one"].append(
                            torch.mean(
                                (
                                    ratio - 1.0 > config.train.clip_range
                                ).float()
                            )
                        )
                        info["clipfrac_lt_one"].append(
                            torch.mean(
                                (
                                    1.0 - ratio > config.train.clip_range
                                ).float()
                            )
                        )
                        info["policy_loss"].append(_safe_info_value(policy_loss))
                        if config.train.beta > 0:
                            info["kl_loss"].append(_safe_info_value(kl_loss))

                        if has_noise_actions:
                            # ExploreNet: clipped PPO objective on the noise actions
                            # plus a KL penalty toward N(0, I).
                            noise_encoder_hidden_states = embeds[embeds.shape[0] // 2:]
                            noise_pooled_text = pooled_embeds[pooled_embeds.shape[0] // 2:]
                            noise_policy_latents = sample["latents"][:, j].detach()
                            noise_log_prob, noise_policy_stats = noise_policy_module(
                                sample["noise_actions"][:, j],
                                noise_policy_latents,
                                sample["timesteps"][:, j],
                                noise_pooled_text,
                                encoder_hidden_states=noise_encoder_hidden_states,
                            )
                            noise_advantages = torch.clamp(
                                sample["noise_advantages"][:, j],
                                -config.train.noise_policy_adv_clip_max,
                                config.train.noise_policy_adv_clip_max,
                            )
                            noise_log_ratio = noise_log_prob - sample["noise_action_log_probs"][:, j]
                            noise_log_ratio = torch.nan_to_num(
                                noise_log_ratio.float(), nan=0.0, posinf=20.0, neginf=-20.0
                            ).clamp(-20.0, 20.0)
                            noise_ratio = torch.exp(noise_log_ratio)
                            noise_unclipped = -noise_advantages * noise_ratio
                            noise_clipped = -noise_advantages * torch.clamp(
                                noise_ratio,
                                1.0 - config.train.noise_policy_clip_range,
                                1.0 + config.train.noise_policy_clip_range,
                            )
                            noise_active = torch.ones_like(noise_log_prob, dtype=torch.float32)
                            noise_active_denom = noise_active.sum().clamp(min=1.0)
                            noise_policy_loss = (
                                torch.maximum(noise_unclipped, noise_clipped) * noise_active
                            ).sum() / noise_active_denom
                            noise_kl_loss = (
                                noise_policy_stats["noise_policy_kl_to_gaussian"] * noise_active
                            ).sum() / noise_active_denom
                            noise_loss = noise_policy_loss + config.train.noise_policy_kl_beta * noise_kl_loss
                            loss = loss + noise_loss
                            info["noise_policy_loss"].append(_safe_info_value(noise_policy_loss))
                            info["noise_policy_kl_loss"].append(_safe_info_value(noise_kl_loss))
                            info["noise_policy_entropy"].append(
                                _safe_info_value(
                                    (noise_policy_stats["noise_policy_entropy"] * noise_active).sum()
                                    / noise_active_denom
                                )
                            )
                            info["noise_policy_ratio_approx_kl"].append(
                                _safe_info_value(
                                    0.5
                                    * ((noise_log_ratio ** 2) * noise_active).sum()
                                    / noise_active_denom
                                )
                            )
                            for stat_key in (
                                "noise_policy_log_std_clamp_min_frac",
                                "noise_policy_log_std_clamp_max_frac",
                                "noise_policy_log_std_channel_mean_std",
                                "noise_policy_log_std_spatial_mean_std",
                            ):
                                info[stat_key].append(
                                    _safe_info_value(
                                        (noise_policy_stats[stat_key] * noise_active).sum()
                                        / noise_active_denom
                                    )
                                )

                        # Skip the update on every rank if any rank produced a nonfinite value.
                        finite_checks = [
                            loss,
                            policy_loss,
                            log_prob,
                            prev_sample_mean,
                            std_dev_t,
                        ]
                        if config.train.beta > 0:
                            finite_checks.extend([kl_loss, prev_sample_mean_ref])
                        if has_noise_actions:
                            finite_checks.extend(
                                [
                                    noise_log_prob,
                                    noise_policy_loss,
                                    noise_kl_loss,
                                    noise_loss,
                                ]
                            )
                            finite_checks.extend(noise_policy_stats.values())
                        local_nonfinite = torch.zeros((), device=accelerator.device, dtype=torch.float32)
                        for value in finite_checks:
                            if isinstance(value, torch.Tensor):
                                local_nonfinite = torch.maximum(
                                    local_nonfinite,
                                    (~torch.isfinite(value.detach()).all()).float(),
                                )
                        global_nonfinite = local_nonfinite.clone()
                        if dist.is_available() and dist.is_initialized():
                            dist.all_reduce(global_nonfinite, op=dist.ReduceOp.MAX)
                        skip_nonfinite = bool(global_nonfinite.item() > 0)
                        info["loss"].append(_safe_info_value(loss))
                        info["nonfinite_loss"].append(global_nonfinite.detach())
                        info["skipped_nonfinite_step"].append(
                            torch.tensor(float(skip_nonfinite), device=accelerator.device)
                        )

                        if skip_nonfinite:
                            optimizer.zero_grad(set_to_none=True)
                            if noise_optimizer is not None:
                                noise_optimizer.zero_grad(set_to_none=True)
                            if accelerator.is_main_process:
                                logger.warning(
                                    f"Epoch {epoch}.{inner_epoch} batch {i} timestep {j}: "
                                    "nonfinite tensor detected; skipped optimizer update on all ranks."
                                )
                        else:
                            # backward pass
                            accelerator.backward(loss)
                            if accelerator.sync_gradients:
                                all_trainable = list(transformer.parameters())
                                if has_noise_actions:
                                    all_trainable += list(noise_policy_module.parameters())
                                accelerator.clip_grad_norm_(
                                    all_trainable, config.train.max_grad_norm
                                )
                            optimizer.step()
                            optimizer.zero_grad()
                            if noise_optimizer is not None and has_noise_actions:
                                noise_optimizer.step()
                                noise_optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        # log training-related stuff
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        if accelerator.is_main_process:
                            wandb.log(info, step=global_step)
                            train_local = {
                                "type": "train",
                                **{k: float(v) if isinstance(v, (torch.Tensor, float, int)) else v for k, v in info.items()},
                            }
                            log_metrics_local(config.save_dir, train_local, global_step)
                        global_step += 1
                        info = defaultdict(list)
                if config.train.ema:
                    ema.step(transformer_trainable_parameters, global_step)

        epoch += 1


if __name__ == "__main__":
    app.run(main)
