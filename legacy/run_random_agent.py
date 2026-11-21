import numpy as np
import pandas as pd

from environment.card_battle_env import CardBattleEnv


def choose_random_affordable_action(env, rng):
    """
    Randomly choose only actions the player can currently afford.
    """

    possible_actions = [env.WAIT]

    knight_cost = env.cards["Knight"]().elixir_cost
    giant_cost = env.cards["Giant"]().elixir_cost

    if env.player_elixir >= knight_cost:
        possible_actions.append(env.KNIGHT_LEFT)
        possible_actions.append(env.KNIGHT_RIGHT)

    if env.player_elixir >= giant_cost:
        possible_actions.append(env.GIANT_LEFT)
        possible_actions.append(env.GIANT_RIGHT)

    return int(rng.choice(possible_actions))


def play_one_episode(seed):
    env = CardBattleEnv(game_duration=120.0)
    env.reset(seed=seed)

    # Makes the player's random choices repeatable for this seed.
    rng = np.random.default_rng(seed)

    steps = 0
    invalid_actions = 0

    while True:
        action = choose_random_affordable_action(env, rng)

        observation, reward, terminated, truncated, info = env.step(action)
        assert observation.shape == (149,)
        assert env.observation_space.contains(observation)

        steps += 1

        if info["invalid_action"]:
            invalid_actions += 1

        if terminated or truncated:
            break

    print(f"\nEpisode seed: {seed}")
    print(f"Winner: {info['winner']}")
    print(f"Final reward: {reward}")
    print(f"Steps: {steps}")
    print(f"Game time used: {steps * env.dt:.1f} seconds")
    print(f"Invalid actions: {invalid_actions}")
    print(f"Player tower HP: {env.player_towers_hp}")
    print(f"Opponent tower HP: {env.opponent_towers_hp}")

    return info["winner"], reward


results = []

for seed in range(1000,1100):
    winner, reward = play_one_episode(seed)
    results.append(winner)

print("\n========== Summary ==========")
print(f"Player wins: {results.count('player')}")
print(f"Opponent wins: {results.count('opponent')}")
print(f"Draws: {results.count('draw')}")
print(f"Total episodes: {len(results)}")
print(f"Player win rate: {results.count('player') / len(results):.2%}")