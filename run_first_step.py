from environment.card_battle_env import CardBattleEnv


env = CardBattleEnv(game_duration=30.0)
env.reset(seed=7)

# 0=wait, 1=Knight-left, 2=Knight-right, 3=Giant-left, 4=Giant-right
actions = (
    [3]          # Giant left: costs 5
    + [0] * 12   # wait until elixir reaches about 3
    + [1]        # Knight left: costs 3
    + [0] * 12   # wait again
    + [2]        # Knight right: costs 3
    + [0] * 80
)
for turn, action in enumerate(actions, start=1):
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"turn={turn}, action={action}, reward={reward}, info={info}")
    env.render()
    print("Player units:")
    for unit in env.player_units:
        print(
            f"  {unit.name} | lane={unit.lane} | "
            f"position={unit.position} | hp={unit.hp}"
        )

    print("Opponent units:")
    for unit in env.opponent_units:
        print(
            f"  {unit.name} | lane={unit.lane} | "
            f"position={unit.position} | hp={unit.hp}"
        )
    if terminated or truncated:
        break
