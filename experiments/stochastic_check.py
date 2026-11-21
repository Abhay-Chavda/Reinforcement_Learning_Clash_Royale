"""Greedy vs sampled actions for agents trained against the random opponent.

Writes results/evaluation/greedy_vs_sampled.csv.
Usage: python -m experiments.stochastic_check
"""

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

from environment.card_battle_env import CardBattleEnv
from experiments.evaluate import EVAL_SEED_START
from experiments.train import GAME_DURATION

SEEDS = (7, 42, 123, 2024, 31337)


def run(job):
    reward, seed, deterministic = job
    model = PPO.load(f"models/ppo_{reward}_vs_random_seed{seed}.zip", device="cpu")
    wins = waits = steps = 0
    for episode_seed in range(EVAL_SEED_START, EVAL_SEED_START + 100):
        torch.manual_seed(episode_seed)
        env = CardBattleEnv(game_duration=GAME_DURATION)
        obs, _ = env.reset(seed=episode_seed)
        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, terminated, truncated, info = env.step(int(action))
            waits += int(action) == 0
            steps += 1
            if terminated or truncated:
                break
        wins += info["winner"] == "player"
    return dict(reward=reward, seed=seed, decoding="greedy" if deterministic else "sampled",
                win_rate=wins / 100, wait_share=waits / steps)


if __name__ == "__main__":
    jobs = [(r, s, d) for r in ("sparse", "shaped") for s in SEEDS for d in (True, False)]
    with ProcessPoolExecutor(max_workers=14) as pool:
        df = pd.DataFrame(pool.map(run, jobs))
    df.to_csv("results/evaluation/greedy_vs_sampled.csv", index=False)
    print(df.groupby(["reward", "decoding"])[["win_rate", "wait_share"]].agg(["mean", "std"]).round(3))
