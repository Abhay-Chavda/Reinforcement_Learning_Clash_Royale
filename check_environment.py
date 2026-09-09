from stable_baselines3.common.env_checker import check_env

from environment.card_battle_env import CardBattleEnv


env = CardBattleEnv()

check_env(
    env,
    warn=True
)

print("Environment check passed.")