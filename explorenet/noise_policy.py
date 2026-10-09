"""ExploreNet: a learned per-element noise policy for Flow-GRPO on SD3.

The policy g_phi is a second SD3 transformer (LoRA-adapted, initialized from
the pretrained denoiser). Its output has the same shape as the latent. For each
latent we standardize the output over all of its elements, clip it to
[log_std_min, log_std_max], and read it as a per-element log standard
deviation. The SDE exploration noise at each denoising step is

    eps = exp(log_std) * z,   z ~ N(0, I),

which then multiplies Flow-GRPO's per-step noise coefficient. The policy is
trained with a clipped PPO-style objective and a KL penalty toward N(0, I);
it is discarded at inference time.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SD3TransformerStdNoisePolicy(nn.Module):
    """Zero-mean diagonal Gaussian policy over the SDE noise of SD3.

    The wrapped transformer is a separate parameter set from the generation
    transformer. Its output is standardized within each latent and used
    directly as a per-element log-std field ("direct_normalized" mode).
    """

    def __init__(
        self,
        transformer: nn.Module,
        log_std_min: float = -1.0,
        log_std_max: float = 1.0,
    ):
        super().__init__()
        self.transformer = transformer
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

    def _positive_embeddings(
        self,
        encoder_hidden_states: torch.Tensor,
        pooled_text: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if encoder_hidden_states is None or pooled_text is None:
            raise ValueError("SD3TransformerStdNoisePolicy requires prompt_embeds and pooled_prompt_embeds")
        if encoder_hidden_states.shape[0] == batch_size * 2:
            encoder_hidden_states = encoder_hidden_states[batch_size:]
        if pooled_text.shape[0] == batch_size * 2:
            pooled_text = pooled_text[batch_size:]
        if encoder_hidden_states.shape[0] != batch_size or pooled_text.shape[0] != batch_size:
            raise ValueError(
                "Prompt embedding batch does not match noise-policy latent batch: "
                f"latents={batch_size}, prompt_embeds={encoder_hidden_states.shape[0]}, pooled={pooled_text.shape[0]}"
            )
        return encoder_hidden_states, pooled_text

    def _raw_transformer_output(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        pooled_text: Optional[torch.Tensor],
        encoder_hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = latents.shape[0]
        encoder_hidden_states, pooled_text = self._positive_embeddings(
            encoder_hidden_states,
            pooled_text,
            batch_size,
        )
        transformer_dtype = next(self.transformer.parameters()).dtype
        out = self.transformer(
            hidden_states=latents.to(dtype=transformer_dtype),
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states.to(dtype=transformer_dtype),
            pooled_projections=pooled_text.to(dtype=transformer_dtype),
            return_dict=False,
        )[0]
        return torch.nan_to_num(out.float(), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)

    def log_std(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        pooled_text: Optional[torch.Tensor],
        encoder_hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        raw = self._raw_transformer_output(latents, timesteps, pooled_text, encoder_hidden_states)
        # Standardize within each latent, then use the result directly as log-std.
        reduce_dims = tuple(range(1, raw.ndim))
        raw = (raw - raw.mean(dim=reduce_dims, keepdim=True)) / raw.std(
            dim=reduce_dims, keepdim=True
        ).clamp(min=1e-6)
        log_std = raw
        log_std = torch.nan_to_num(
            log_std,
            nan=0.0,
            posinf=self.log_std_max,
            neginf=self.log_std_min,
        ).clamp(self.log_std_min, self.log_std_max)
        return torch.nan_to_num(
            log_std,
            nan=0.0,
            posinf=self.log_std_max,
            neginf=self.log_std_min,
        ).clamp(self.log_std_min, self.log_std_max)

    @staticmethod
    def diagonal_zero_mean_log_prob(action: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        log_prob = -0.5 * (action.float() / log_std.exp()) ** 2 - log_std - 0.5 * math.log(2 * math.pi)
        return log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    def log_std_structure_stats(self, log_std: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Per-sample logging summaries of the log-std field (no effect on training)."""
        x = log_std.float()
        batch = x.shape[0]
        flat = x.flatten(1)
        channels = x.shape[1]
        channel_means = x.reshape(batch, channels, -1).mean(dim=-1)
        spatial_map = x.mean(dim=1).reshape(batch, -1)
        return {
            "noise_policy_log_std_clamp_min_frac": (flat <= self.log_std_min + 1e-6).float().mean(dim=1),
            "noise_policy_log_std_clamp_max_frac": (flat >= self.log_std_max - 1e-6).float().mean(dim=1),
            "noise_policy_log_std_channel_mean_std": channel_means.std(dim=1),
            "noise_policy_log_std_spatial_mean_std": spatial_map.std(dim=1),
        }

    def log_prob(
        self,
        action: torch.Tensor,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        pooled_text: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        log_std = self.log_std(latents, timesteps, pooled_text, encoder_hidden_states)
        log_prob = self.diagonal_zero_mean_log_prob(action.float(), log_std)
        reduce_dims = tuple(range(1, log_std.ndim))
        entropy = (log_std + 0.5 * (1.0 + math.log(2 * math.pi))).mean(dim=reduce_dims)
        # KL(N(0, exp(log_std)^2) || N(0, 1)), averaged over latent elements.
        kl = 0.5 * (torch.exp(2.0 * log_std) - 1.0 - 2.0 * log_std)
        stats = {
            "noise_policy_entropy": entropy,
            "noise_policy_kl_to_gaussian": kl.mean(dim=reduce_dims),
            "noise_policy_log_std_mean": log_std.mean(dim=reduce_dims),
            "noise_policy_log_std_std": log_std.flatten(1).std(dim=1),
        }
        stats.update(self.log_std_structure_stats(log_std))
        return log_prob.float(), stats

    def forward(
        self,
        action: torch.Tensor,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        pooled_text: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.log_prob(action, latents, timesteps, pooled_text, encoder_hidden_states)

    @torch.no_grad()
    def sample_mixed_noise(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        pooled_text: Optional[torch.Tensor],
        alpha: float,
        base_noise: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Sample the policy action and the SDE noise it induces.

        Returns ``eps_mix = sqrt(1 - alpha^2) * base_noise + alpha * action``.
        The paper uses ``alpha = 1``, so the SDE noise equals the policy action
        ``exp(log_std) * z``.
        """
        log_std = self.log_std(latents, timesteps, pooled_text, encoder_hidden_states)
        raw_action = log_std.exp() * torch.randn_like(log_std)
        raw_action = torch.nan_to_num(raw_action, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
        log_prob = self.diagonal_zero_mean_log_prob(raw_action, log_std)
        eps_net = raw_action.float()
        eps_mix = math.sqrt(max(0.0, 1.0 - alpha ** 2)) * base_noise.float() + alpha * eps_net
        eps_mix = torch.nan_to_num(eps_mix, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
        reduce_dims = tuple(range(1, eps_mix.ndim))
        stats = {
            "noise_eps_net_std": eps_net.std(dim=reduce_dims),
            "noise_eps_mix_std": eps_mix.std(dim=reduce_dims),
            "noise_abs_mix_minus_base": (eps_mix - base_noise.float()).abs().mean(dim=reduce_dims),
            "noise_cos_base_net": F.cosine_similarity(base_noise.flatten(1).float(), eps_net.flatten(1).float(), dim=1),
            "noise_policy_log_std_mean": log_std.mean(dim=reduce_dims),
            "noise_policy_log_std_std": log_std.flatten(1).std(dim=1),
        }
        stats.update(self.log_std_structure_stats(log_std))
        return eps_mix.float(), raw_action.float(), log_prob.float(), stats

    def state_dict(self, *args, **kwargs):
        # Save only the trainable (LoRA) parameters; the frozen base transformer
        # is reloaded from the pretrained checkpoint.
        state = super().state_dict(*args, **kwargs)
        trainable_names = {name for name, param in self.named_parameters() if param.requires_grad}
        return {
            key: value
            for key, value in state.items()
            if key in trainable_names
        }

    def load_state_dict(self, state_dict, strict: bool = False):
        return super().load_state_dict(state_dict, strict=False)
