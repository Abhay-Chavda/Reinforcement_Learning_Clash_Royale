"""Evaluate every trained agent and baseline against every evaluation opponent.

Writes one row per episode to ``results/evaluation/episodes.csv``.
Usage: python -m experiments.evaluate [--episodes 100] [--workers 8]
"""

import argparse
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from environment.card_battle_env import CardBattleEnv
from experiments.agents import HeuristicAgent, RandomAgent
from experiments.train import DEFAULT_OPPONENT_MODEL, GAME_DURATION

EVAL_SEED_START = 1000  # held out: training never resets with these seeds
EVAL_OPPONENTS = ("random", "heuristic", "ppo_frozen")
NAME_RE = re.compile(r"ppo_(?P<reward>sparse|shaped)_vs_(?P<train_opp>\w+?)_seed(?P<seed>\d+)$")


def _load_agent(agent_id, episode_seed):
    if agent_id == "random":
        return RandomAgent(seed=episode_seed)
    if agent_id == "heuristic":
        return HeuristicAgent()
    from stable_baselines3 import PPO
    return PPO.load(f"models/{agent_id}.zip", device="cpu")


def _opponent_policy(opponent_id):
    if opponent_id == "random":
        return None
    if opponent_id == "heuristic":
        return HeuristicAgent().predict
    from stable_baselines3 import PPO
    return PPO.load(DEFAULT_OPPONENT_MODEL, device="cpu").predict


def evaluate_pair(job):
    agent_id, opponent_id, episodes = job
    opponent = _opponent_policy(opponent_id)
    model = None if agent_id in ("random", "heuristic") else _load_agent(agent_id, 0)
    rows = []
    for episode_seed in range(EVAL_SEED_START, EVAL_SEED_START + episodes):
        agent = model or _load_agent(agent_id, episode_seed)
        env = CardBattleEnv(game_duration=GAME_DURATION, opponent_policy=opponent)
        obs, _ = env.reset(seed=episode_seed)
        actions = np.zeros(5, dtype=int)
        invalid = steps = 0
        while True:
            action, _ = agent.predict(obs, deterministic=True)
            action = int(action)
            obs, _, terminated, truncated, info = env.step(action)
            actions[action] += 1
            invalid += info["invalid_action"]
            steps += 1
            if terminated or truncated:
                break
        rows.append(dict(
            agent=agent_id, eval_opponent=opponent_id, episode_seed=episode_seed,
            winner=info["winner"], king_destroyed=bool(terminated), steps=steps,
            player_tower_hp=float(env.player_towers_hp.sum()),
            opponent_tower_hp=float(env.opponent_towers_hp.sum()),
            invalid_actions=invalid,
            **{f"act_{n}": int(c) for n, c in zip(
                ["wait", "knight_left", "knight_right", "giant_left", "giant_right"], actions)},
        ))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    models = sorted(p.stem for p in Path("models").glob("ppo_*.zip") if NAME_RE.match(p.stem))
    agents = ["random", "heuristic"] + models
    jobs = [(a, o, args.episodes) for a in agents for o in EVAL_OPPONENTS]
    print(f"Evaluating {len(agents)} agents x {len(EVAL_OPPONENTS)} opponents x {args.episodes} episodes")

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = [row for result in pool.map(evaluate_pair, jobs) for row in result]

    df = pd.DataFrame(rows)
    parsed = df["agent"].str.extract(NAME_RE)
    df["reward"] = parsed["reward"].fillna("-")
    df["train_opponent"] = parsed["train_opp"].fillna("-")
    df["train_seed"] = pd.to_numeric(parsed["seed"])
    df["condition"] = np.where(
        parsed["reward"].isna(), df["agent"].str.capitalize() + " baseline",
        "PPO " + df["reward"] + " / train vs " + df["train_opponent"])
    Path("results/evaluation").mkdir(parents=True, exist_ok=True)
    df.to_csv("results/evaluation/episodes.csv", index=False)
    print(df.groupby(["condition", "eval_opponent"])["winner"]
            .apply(lambda w: (w == "player").mean()).unstack().round(3))


if __name__ == "__main__":
    main()
