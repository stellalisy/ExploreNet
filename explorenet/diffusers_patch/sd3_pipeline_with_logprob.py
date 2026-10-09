# Adapted from Flow-GRPO (https://github.com/yifan123/flow_grpo), which copies
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py
# with the following modifications:
# - It uses the patched version of `sde_step_with_logprob` from `sd3_sde_with_logprob.py`.
# - It returns all the intermediate latents of the denoising process as well as the log probs of each denoising step.
# ExploreNet adds an optional learned noise policy that sets the SDE noise of each step.
from typing import Any, Dict, List, Optional, Union

import torch
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps

from .sd3_sde_with_logprob import sample_gaussian_noise, sde_step_with_logprob


@torch.no_grad()
def pipeline_with_logprob(
    self,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    guidance_scale: float = 7.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    max_sequence_length: int = 256,
    noise_level: float = 0.7,
    learned_noise_module=None,
    learned_noise_alpha: float = 0.0,
):
    """Sample images with Flow-GRPO's SDE sampler and record per-step log-probs.

    If ``learned_noise_module`` is given (ExploreNet), the SDE noise of every
    step is drawn from the learned noise policy, and the policy's actions and
    log-probs are returned as a fourth output.
    """
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        None,
        None,
        None,
        height,
        width,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=["latents"],
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._clip_skip = None
    self._joint_attention_kwargs = joint_attention_kwargs
    self._interrupt = False

    # 2. Define call parameters
    batch_size = prompt_embeds.shape[0]
    device = self._execution_device

    lora_scale = (
        self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
    )
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = self.encode_prompt(
        prompt=None,
        prompt_2=None,
        prompt_3=None,
        do_classifier_free_guidance=self.do_classifier_free_guidance,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        device=device,
        clip_skip=self.clip_skip,
        num_images_per_prompt=1,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )
    if self.do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

    # 4. Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels
    latents = self.prepare_latents(
        batch_size,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        None,
    ).float()

    # 5. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        sigmas=None,
    )
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    self._num_timesteps = len(timesteps)

    all_latents = [latents]
    all_log_probs = []
    all_noise_actions = []
    all_noise_log_probs = []
    all_noise_stats = []
    use_learned_noise = learned_noise_module is not None and learned_noise_alpha > 0

    # 7. Denoising loop
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            latent_model_input = latent_model_input.to(dtype=prompt_embeds.dtype)
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latent_model_input.shape[0])

            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,
            )[0]
            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            learned_variance_noise = None
            if use_learned_noise:
                # Draw the Flow-GRPO Gaussian noise first; with alpha = 1 it is
                # fully replaced by the policy action, but drawing it keeps the
                # random stream aligned with the reference implementation.
                base_noise = sample_gaussian_noise(
                    noise_pred.shape, generator, noise_pred.device, noise_pred.dtype,
                )
                if self.do_classifier_free_guidance:
                    noise_pooled_text = pooled_prompt_embeds[pooled_prompt_embeds.shape[0] // 2:]
                    noise_encoder_hidden_states = prompt_embeds[prompt_embeds.shape[0] // 2:]
                else:
                    noise_pooled_text = pooled_prompt_embeds
                    noise_encoder_hidden_states = prompt_embeds
                learned_variance_noise, noise_action, noise_log_prob, noise_stats = learned_noise_module.sample_mixed_noise(
                    latents=latents.float(),
                    timesteps=t.expand(latents.shape[0]),
                    pooled_text=noise_pooled_text,
                    alpha=learned_noise_alpha,
                    base_noise=base_noise.float(),
                    encoder_hidden_states=noise_encoder_hidden_states,
                )
                all_noise_actions.append(noise_action)
                all_noise_log_probs.append(noise_log_prob)
                all_noise_stats.append(noise_stats)

            latents, log_prob, _, _ = sde_step_with_logprob(
                self.scheduler,
                noise_pred.float(),
                t.unsqueeze(0),
                latents.float(),
                noise_level=noise_level,
                variance_noise=learned_variance_noise,
            )

            all_latents.append(latents)
            all_log_probs.append(log_prob)

            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

    latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
    latents = latents.to(dtype=self.vae.dtype)
    image = self.vae.decode(latents, return_dict=False)[0]
    image = self.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    self.maybe_free_model_hooks()

    if use_learned_noise:
        return image, all_latents, all_log_probs, {
            "noise_actions": all_noise_actions,
            "noise_log_probs": all_noise_log_probs,
            "noise_stats": all_noise_stats,
        }
    return image, all_latents, all_log_probs
