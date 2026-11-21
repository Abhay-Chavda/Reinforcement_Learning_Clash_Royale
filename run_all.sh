#!/usr/bin/env bash
# Reproduce every experiment in the paper (~15 min on a 16-thread CPU).
set -euo pipefail
SEEDS="7 42 123 2024 31337"
TIMESTEPS=${TIMESTEPS:-100000}

# Stage 1: PPO vs the random opponent (sparse and shaped reward).
for reward in sparse shaped; do
  for seed in $SEEDS; do
    python -m experiments.train --reward $reward --opponent random --seed $seed --timesteps $TIMESTEPS &
  done
done
wait

# Stage 2: PPO vs a frozen PPO opponent (the stage-1 sparse seed-7 agent).
for reward in sparse shaped; do
  for seed in $SEEDS; do
    python -m experiments.train --reward $reward --opponent ppo --seed $seed --timesteps $TIMESTEPS \
      --opponent-model models/ppo_sparse_vs_random_seed7.zip &
  done
done
wait

python -m experiments.evaluate
python -m experiments.analyze
