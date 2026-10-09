# Adapted from Flow-GRPO (https://github.com/yifan123/flow_grpo), which copies
# https://github.com/kvablack/ddpo-pytorch/blob/main/ddpo_pytorch/diffusers_patch/ddim_with_logprob.py
# and adapts it from diffusion to flow matching.

import math
from typing import Optional, Union

import torch
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor


def sample_gaussian_noise(shape, generator, device, dtype):
    """Standard Gaussian SDE noise (the Flow-GRPO default)."""
    return randn_tensor(shape, generator=generator, device=device, dtype=dtype)


def sde_step_with_logprob(
    self: FlowMatchEulerDiscreteScheduler,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    noise_level: float = 0.7,
    prev_sample: Optional[torch.FloatTensor] = None,
    generator: Optional[torch.Generator] = None,
    variance_noise: Optional[torch.FloatTensor] = None,
):
    """One Flow-GRPO SDE step and the Gaussian log-prob of its transition.

    Args:
        model_output: predicted velocity from the flow model.
        timestep: current discrete timestep(s).
        sample: current latent x_t.
        noise_level: Flow-GRPO noise scale (``a`` in sigma_t = a * sqrt(t / (1 - t))).
        prev_sample: if given, the log-prob of this next latent is computed and
            no noise is drawn (training-time replay).
        variance_noise: optional pre-drawn noise for the step. ExploreNet passes
            ``exp(log_std) * z`` here, which rescales Flow-GRPO's noise per
            latent element. The log-prob below always uses the scalar
            Flow-GRPO std, so the denoiser keeps the Flow-GRPO objective.
    """
    # bf16 can overflow here when computing prev_sample_mean, so convert everything to fp32
    sample = sample.float()
    model_output = model_output.to(device=sample.device, dtype=sample.dtype)
    if prev_sample is not None:
        prev_sample = prev_sample.to(device=sample.device, dtype=sample.dtype)

    timestep_for_index = timestep.to(self.timesteps.device)
    step_index = [self.index_for_timestep(t) for t in timestep_for_index]
    prev_step_index = [step + 1 for step in step_index]
    sigma = self.sigmas[step_index].to(device=sample.device, dtype=sample.dtype).view(
        -1, *([1] * (len(sample.shape) - 1))
    )
    sigma_prev = self.sigmas[prev_step_index].to(device=sample.device, dtype=sample.dtype).view(
        -1, *([1] * (len(sample.shape) - 1))
    )
    sigma_max = self.sigmas[1].to(device=sample.device, dtype=sample.dtype)
    dt = sigma_prev - sigma

    sigma_denom = torch.where(
        sigma == 1,
        sigma_max.view(*([1] * sigma.ndim)),
        sigma,
    )
    std_dev_t_scalar = torch.sqrt(sigma / (1 - sigma_denom)) * noise_level

    prev_sample_mean = sample * (1 + std_dev_t_scalar**2 / (2 * sigma) * dt) + model_output * (1 + std_dev_t_scalar**2 * (1 - sigma) / (2 * sigma)) * dt

    if prev_sample is None:
        if variance_noise is None:
            variance_noise = sample_gaussian_noise(
                model_output.shape, generator, model_output.device, model_output.dtype,
            )
        else:
            variance_noise = variance_noise.to(device=model_output.device, dtype=model_output.dtype)
        noise_std = std_dev_t_scalar
        prev_sample = prev_sample_mean + noise_std * torch.sqrt(-1 * dt) * variance_noise

    scale = std_dev_t_scalar * torch.sqrt(-1 * dt)
    z = (prev_sample.detach() - prev_sample_mean) / scale
    log_prob = (-0.5 * z ** 2 - 0.5 * math.log(2 * math.pi)) - torch.log(scale)

    # mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    return prev_sample, log_prob, prev_sample_mean, std_dev_t_scalar
