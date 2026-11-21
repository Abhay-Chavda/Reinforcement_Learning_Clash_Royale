from environment.card_battle_env import CardBattleEnv

env = CardBattleEnv()

observation, info = env.reset()

print("==================Initail Observation==================")

print(
    "Player:",
    env.player_units[0].name,
    "Positions:",
    env.player_units[0].position,
    "HP:",
    env.player_units[0].hp
)

print(
    "opponent:Tower",
    "Positions:",
    env.opponent_towers_position,
    "HP:",
    env.opponent_towers_hp
)

print("\n==============BATTLE==============")

for step in range(100):
    observation,reward, terminated, truncated,info = env.step(0)

    print(f"\n Step {step+1}")


    print(
        "Player Giant:",
        "Position = ", env.player_units[0].position,
        "HP = ",env.player_units[0].hp
    )

    print(
        "Opponent tower:",
        "Position = ",env.opponent_towers_position,
        "HP = ",env.opponent_towers_hp
    )

    if not env.player_units[0].alive:
        print("Player Giant Died !")
        break