# Adapted from Flow-GRPO (https://github.com/yifan123/flow_grpo).
"""GenEval2 prompt dataset and the K-repeat distributed prompt sampler."""

import json
import math
import os

import torch
from torch.utils.data import Dataset, Sampler


class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}_metadata.jsonl')
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item['prompt'] for item in self.metadatas]
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0, rollout_batches=1):
        self.dataset = dataset
        self.batch_size = batch_size  # Batch size per replica
        self.k = k                    # Number of repetitions per sample
        self.num_replicas = num_replicas  # Total number of replicas
        self.rank = rank              # Current replica rank
        self.seed = seed              # Random seed for synchronization
        self.rollout_batches = max(1, int(rollout_batches))
        
        # Compute the number of unique samples needed per iteration
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, f"k can not divide n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k  # Number of unique samples
        self.epoch = 0

    def _permutation_for_cycle(self, population_size, cycle):
        g = torch.Generator()
        g.manual_seed(self.seed + cycle)
        return torch.randperm(population_size, generator=g).tolist(), g

    def _selected_unique_indices(self, population):
        """Return the unique prompt rows for this sampler step.

        Training checks prompt uniqueness over the whole rollout, not only one
        per-rank batch.  When a rollout crosses the boundary between two
        shuffled passes through a small active pool, taking the tail of one
        permutation plus the head of the next can repeat a row inside the same
        rollout.  Build the rollout-sized block first and filter the next
        permutation against the tail before slicing this sampler step.
        """
        population_size = len(population)
        rollout_m = self.m * self.rollout_batches
        if rollout_m > population_size:
            raise ValueError(
                "DistributedKRepeatSampler rollout requires more unique samples "
                f"than the active population provides: rollout_unique={rollout_m} "
                f"population={population_size}"
            )

        rollout_epoch = self.epoch // self.rollout_batches
        batch_in_rollout = self.epoch % self.rollout_batches
        rollout_cycle_len = max(1, math.ceil(population_size / rollout_m))
        cycle = rollout_epoch // rollout_cycle_len
        offset = (rollout_epoch * rollout_m) % population_size

        order, g = self._permutation_for_cycle(population_size, cycle)
        if offset + rollout_m <= population_size:
            block = order[offset:offset + rollout_m]
        else:
            remaining = population_size - offset
            block = order[offset:]
            used = set(block)
            next_cycle = cycle + 1
            while len(block) < rollout_m:
                next_order, _ = self._permutation_for_cycle(population_size, next_cycle)
                block.extend(idx for idx in next_order if idx not in used)
                block = block[:rollout_m]
                used = set(block)
                next_cycle += 1

        start = batch_in_rollout * self.m
        selected = block[start:start + self.m]
        return [population[idx] for idx in selected], g

    def __iter__(self):
        while True:
            # Generate a deterministic random sequence to ensure all replicas are synchronized
            population = list(range(len(self.dataset)))
            if not population:
                raise ValueError("DistributedKRepeatSampler population cannot be empty")

            indices, g = self._selected_unique_indices(population)

            # Repeat each sample k times to generate n*b total samples
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            
            # Shuffle to ensure uniform distribution
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            
            # Split samples to each replica
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            
            # Return current replica's sample indices
            yield per_card_samples[self.rank]
    
    def set_epoch(self, epoch):
        self.epoch = epoch  # Used to synchronize random state across epochs

