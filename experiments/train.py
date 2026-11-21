"""Train one PPO agent.

Examples
--------
python -m experiments.train --reward sparse --opponent random --seed 7
python -m experiments.train --reward shaped --opponent ppo --seed 42 \
    --opponent-model models/ppo_sparse_vs_random_seed7.zip
"""

import argparse
import json
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from environment.card_battle_env import CardBattleEnv
from experiments.agents import HeuristicAgent

GAME_DURATION = 120.0
PPO_KWARGS = dict(
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
)
DEFAULT_OPPONENT_MODEL = "models/ppo_sparse_vs_random_seed7.zip"


def run_name(reward, opponent, seed):
    return f"ppo_{reward}_vs_{opponent}_seed{seed}"


def make_opponent(opponent, opponent_model):
    if opponent == "random":
        return None  # the environment's built-in random opponent
    if opponent == "heuristic":
        return HeuristicAgent().predict
    if opponent == "ppo":
        return PPO.load(opponent_model, device="cpu").predict
    raise ValueError(opponent)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward", choices=["sparse", "shaped"], required=True)
    parser.add_argument("--opponent", choices=["random", "heuristic", "ppo"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--opponent-model", default=DEFAULT_OPPONENT_MODEL)
    args = parser.parse_args()

    name = run_name(args.reward, args.opponent, args.seed)
    for folder in ("models", "logs", "results/train_monitor"):
        Path(folder).mkdir(parents=True, exist_ok=True)

    env = CardBattleEnv(
        game_duration=GAME_DURATION,
        reward_shaping=args.reward == "shaped",
        opponent_policy=make_opponent(args.opponent, args.opponent_model),
    )
    env = Monitor(env, f"results/train_monitor/{name}", info_keywords=("winner",))
    env.reset(seed=args.seed)

    model = PPO("MlpPolicy", env, seed=args.seed, device="cpu", verbose=0,
                tensorboard_log="logs/", **PPO_KWARGS)
    start = time.time()
    model.learn(total_timesteps=args.timesteps, tb_log_name=name, progress_bar=False)
    model.save(f"models/{name}")

    meta = dict(vars(args), name=name, wall_clock_seconds=round(time.time() - start, 1),
                game_duration=GAME_DURATION, ppo=PPO_KWARGS)
    Path(f"models/{name}.json").write_text(json.dumps(meta, indent=2))
    print(f"[{name}] done in {meta['wall_clock_seconds']} s")


if __name__ == "__main__":
    main()
