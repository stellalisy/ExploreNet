"""Paper configurations: Flow-GRPO and ExploreNet on SD3.5-M with the GenEval2 reward.

Select a config with ``--config=config/explorenet_sd35.py:<name>[,num_gpus=<N>]``:

    flowgrpo_sd35_geneval2      Flow-GRPO baseline
    explorenet_sd35_geneval2    ExploreNet (Flow-GRPO + learned noise policy)

The two configs differ only in the ``noise_policy_*`` keys and
``sample.use_learned_noise_policy``. The paper runs used 64 GPUs. With
``num_gpus=N`` (N a multiple of 8, dividing 512), the per-iteration batch
(1,536 rollouts = 64 prompts x 24 images) and the two optimizer updates per
iteration are kept by scaling ``num_batches_per_epoch`` and
``gradient_accumulation_steps``; the sampled prompts and random streams then
differ from a 64-GPU run.

Other fields can be overridden on the command line, e.g. ``--config.seed=123``
(the paper used seeds 42, 123 and 456) or ``--config.save_dir=...``.
"""

import os

import ml_collections

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PAPER_NUM_GPUS = 64


def _base_config():
    config = ml_collections.ConfigDict()

    ###### General ######
    # run name for wandb logging and checkpoint saving (a timestamp is appended).
    config.run_name = ""
    config.seed = 42
    # number of processes the batch schedule below is built for.
    config.num_gpus = _PAPER_NUM_GPUS
    # number of epochs (sampling iterations) between checkpoints / evaluations.
    config.save_freq = 30
    config.eval_freq = 30
    # stop after this many optimizer updates (None: run until stopped). The
    # paper's main results use the checkpoint at update 3000.
    config.max_global_step = ml_collections.config_dict.placeholder(int)
    # mixed precision for the frozen modules (the LoRA weights stay in fp32).
    config.mixed_precision = "fp16"
    config.allow_tf32 = True
    config.dataset = os.path.join(_REPO_ROOT, "dataset/geneval2")
    config.resolution = 512
    config.save_dir = ""

    ###### Pretrained Model ######
    config.pretrained = pretrained = ml_collections.ConfigDict()
    pretrained.model = "stabilityai/stable-diffusion-3.5-medium"

    ###### Sampling ######
    config.sample = sample = ml_collections.ConfigDict()
    # number of sampler inference steps for collecting rollouts.
    sample.num_steps = 25
    # number of sampler inference steps for evaluation.
    sample.eval_num_steps = 25
    # classifier-free guidance weight.
    sample.guidance_scale = 4.5
    # rollouts per GPU per sampling batch.
    sample.train_batch_size = 3
    # group size: images per prompt.
    sample.num_image_per_prompt = 24
    sample.test_batch_size = 14
    # sampling batches per epoch; rollouts per epoch = num_batches_per_epoch * train_batch_size * num_gpus.
    sample.num_batches_per_epoch = 8
    # normalize advantages by the std of all rewards in the iteration.
    sample.global_std = True
    # Flow-GRPO SDE noise level a (sigma_t = a * sqrt(t / (1 - t))).
    sample.noise_level = 0.7
    # ExploreNet learned noise policy.
    sample.use_learned_noise_policy = False

    ###### Training ######
    config.train = train = ml_collections.ConfigDict()
    # batch size (per GPU) for the policy update.
    train.batch_size = 3
    train.learning_rate = 3e-4
    train.adam_beta1 = 0.9
    train.adam_beta2 = 0.999
    train.adam_weight_decay = 1e-4
    train.adam_epsilon = 1e-8
    # number of training batches accumulated per optimizer update.
    train.gradient_accumulation_steps = 4
    train.max_grad_norm = 1.0
    train.num_inner_epochs = 1
    # clip advantages to the range [-adv_clip_max, adv_clip_max].
    train.adv_clip_max = 5
    # the PPO clip range.
    train.clip_range = 1e-4
    # train on int(num_steps * timestep_fraction) = 24 of the 25 steps.
    train.timestep_fraction = 0.99
    # KL coefficient toward the pretrained model.
    train.beta = 0.04
    # LoRA checkpoint (".../checkpoint-N/lora") to resume from.
    train.lora_path = ml_collections.config_dict.placeholder(str)
    # evaluate and save with EMA weights.
    train.ema = True

    ###### Reward ######
    config.reward_fn = {"geneval2": 1.0}
    return config


def _add_explorenet(config):
    sample = config.sample
    train = config.train
    sample.use_learned_noise_policy = True
    # g_phi: a second SD3.5-M transformer with its own LoRA, in fp32.
    sample.noise_policy_lora_rank = 32
    sample.noise_policy_lora_alpha = 64
    # clip range of the standardized log-std field.
    sample.noise_policy_log_std_min = -1.0
    sample.noise_policy_log_std_max = 1.0
    # SDE noise = sqrt(1 - alpha^2) * N(0, I) + alpha * exp(log_std) * z; the paper uses alpha = 1.
    sample.noise_policy_alpha = 1.0
    train.noise_policy_learning_rate = 1e-4
    train.noise_policy_weight_decay = 0.0
    train.noise_policy_clip_range = 1e-4
    train.noise_policy_adv_clip_max = 5
    # beta_phi: KL penalty toward N(0, I).
    train.noise_policy_kl_beta = 1e-3
    return config


def _set_num_gpus(config, num_gpus):
    num_gpus = int(num_gpus)
    total_batches = _PAPER_NUM_GPUS * config.sample.num_batches_per_epoch
    total_accum = _PAPER_NUM_GPUS * config.train.gradient_accumulation_steps
    if num_gpus <= 0 or total_batches % num_gpus or total_accum % num_gpus:
        raise ValueError(f"num_gpus={num_gpus} must divide {total_batches} and {total_accum}")
    if (num_gpus * config.sample.train_batch_size) % config.sample.num_image_per_prompt:
        raise ValueError(
            f"num_gpus * train_batch_size must be divisible by num_image_per_prompt="
            f"{config.sample.num_image_per_prompt}"
        )
    config.num_gpus = num_gpus
    config.sample.num_batches_per_epoch = total_batches // num_gpus
    config.train.gradient_accumulation_steps = total_accum // num_gpus
    return config


def flowgrpo_sd35_geneval2():
    config = _base_config()
    config.run_name = "flowgrpo_sd35_geneval2"
    config.save_dir = os.path.join("logs", config.run_name)
    return config


def explorenet_sd35_geneval2():
    config = _add_explorenet(_base_config())
    config.run_name = "explorenet_sd35_geneval2"
    config.save_dir = os.path.join("logs", config.run_name)
    return config


def get_config(name):
    """Parse ``<name>[,num_gpus=<N>]``."""
    parts = [part.strip() for part in str(name).split(",") if part.strip()]
    builders = {
        "flowgrpo_sd35_geneval2": flowgrpo_sd35_geneval2,
        "explorenet_sd35_geneval2": explorenet_sd35_geneval2,
    }
    if not parts or parts[0] not in builders:
        raise ValueError(f"Unknown config {name!r}; choose one of {sorted(builders)}")
    config = builders[parts[0]]()
    for option in parts[1:]:
        key, _, value = option.partition("=")
        if key != "num_gpus":
            raise ValueError(f"Unknown config option {option!r}; only num_gpus=<N> is supported")
        _set_num_gpus(config, value)
    return config
