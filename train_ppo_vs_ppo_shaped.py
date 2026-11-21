from pathlib import Path

from stable_baselines3 import PPO

from environment.card_battle_env import CardBattleEnv


SEED = 7
TOTAL_TIMESTEPS = 100_000
MODEL_NAME = f"ppo_vs_ppo_shaped_seed_{SEED}"


# Create folders for saved research artifacts.
Path("models").mkdir(exist_ok=True)
Path("logs").mkdir(exist_ok=True)

# This environment contains the fixed random opponent.
opponent = PPO.load("models/ppo_sparse_seed_7.zip")
env = CardBattleEnv(game_duration=120.0, reward_shaping=True, opponent_policy=opponent.predict)
env.reset(seed=SEED)

model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1,
    seed=SEED,
    device="cpu",
    tensorboard_log="logs/"
)

print("Starting PPO training...")

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    progress_bar=False
)

model.save(f"models/{MODEL_NAME}")

print("Training complete.")
print(f"Model saved to: models/{MODEL_NAME}.zip")