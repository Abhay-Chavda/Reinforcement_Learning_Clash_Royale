import csv
from pathlib import Path

from stable_baselines3 import PPO

from environment.card_battle_env import CardBattleEnv


MODEL_PATH = "models/ppo_sparse_seed_123.zip"
EVALUATION_SEEDS = range(1000, 1100)

Path("results").mkdir(exist_ok=True)

model = PPO.load(MODEL_PATH)

results = []

for seed in EVALUATION_SEEDS:
    env = CardBattleEnv(game_duration=120.0)
    observation, info = env.reset(seed=seed)

    steps = 0

    while True:
        action, _ = model.predict(
            observation,
            deterministic=True
        )

        observation, reward, terminated, truncated, info = env.step(
            int(action)
        )

        steps += 1

        if terminated or truncated:
            break

    results.append({
        "seed": seed,
        "winner": info["winner"],
        "reward": reward,
        "steps": steps,
        "player_tower_hp": env.player_towers_hp.sum(),
        "opponent_tower_hp": env.opponent_towers_hp.sum(),
    })

    print(
        f"Seed {seed}: "
        f"winner={info['winner']}, "
        f"reward={reward}"
    )

with open(
    "results/ppo_evaluation_seed_123.csv",
    "w",
    newline=""
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=results[0].keys()
    )
    writer.writeheader()
    writer.writerows(results)

player_wins = sum(
    result["winner"] == "player"
    for result in results
)

opponent_wins = sum(
    result["winner"] == "opponent"
    for result in results
)

draws = sum(
    result["winner"] == "draw"
    for result in results
)

total_games = len(results)

print("\n========== PPO Evaluation Summary ==========")
print(f"Player wins: {player_wins}")
print(f"Opponent wins: {opponent_wins}")
print(f"Draws: {draws}")
print(f"Total episodes: {total_games}")
print(f"PPO player win rate: {player_wins / total_games:.2%}")
print("Saved: results/ppo_sparse_seed_7.csv")