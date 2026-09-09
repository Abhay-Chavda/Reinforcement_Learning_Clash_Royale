import numpy as np

from environment.card_battle_env import CardBattleEnv
from environment.card_library.cards import Knight
from environment.combat import attack_unit
from environment.unit import Unit


def test_reset_creates_clean_game():
    env = CardBattleEnv()
    observation, info = env.reset(seed=7)

    assert observation.shape == (149,)
    assert env.observation_space.contains(observation)

    assert len(env.player_units) == 0
    assert len(env.opponent_units) == 0

    np.testing.assert_array_equal(
        env.player_towers_hp,
        np.array([2500.0, 5000.0, 2500.0])
    )

    assert env.player_elixir == 5.0
    assert env.opponent_elixir == 5.0
    assert env.time_remaining == env.game_duration


def test_giant_deployment_uses_elixir():
    env = CardBattleEnv()
    env.reset(seed=7)

    # Action 3 = Giant left
    _, _, _, _, info = env.step(3)

    assert info["invalid_action"] is False
    assert len(env.player_units) == 1
    assert env.player_units[0].name == "Giant"

    # Giant costs all starting elixir. It cannot deploy again immediately.
    _, _, _, _, info = env.step(3)

    assert info["invalid_action"] is True
    assert len(env.player_units) == 1


def test_attack_cooldown_prevents_instant_second_attack():
    knight_card = Knight()

    attacker = Unit(
        troop_type=knight_card,
        position=[0.25, 0.50],
        lane="left",
        team="player"
    )

    target = Unit(
        troop_type=knight_card,
        position=[0.25, 0.50],
        lane="left",
        team="opponent"
    )

    starting_hp = target.hp

    assert attack_unit(attacker, target) is True
    assert target.hp == starting_hp - attacker.damage

    # The attacker should still be on cooldown.
    assert attack_unit(attacker, target) is False
    assert target.hp == starting_hp - attacker.damage

    # Advance exactly one full cooldown.
    attacker.update_cooldown(attacker.attack_cooldown)

    assert attack_unit(attacker, target) is True
    assert target.hp == starting_hp - (2 * attacker.damage)


def test_troops_target_queen_before_king():
    env = CardBattleEnv()
    env.reset(seed=7)

    # Alive Queen towers are selected first.
    assert env.get_target_tower("left", env.opponent_towers_hp) == 0
    assert env.get_target_tower("right", env.opponent_towers_hp) == 2

    # When the left Queen tower is destroyed, left troops target King.
    env.opponent_towers_hp[0] = 0
    assert env.get_target_tower("left", env.opponent_towers_hp) == 1

    # If Queen and King are both destroyed, no valid target remains.
    env.opponent_towers_hp[1] = 0
    assert env.get_target_tower("left", env.opponent_towers_hp) is None


def test_destroyed_king_tower_ends_match():
    env = CardBattleEnv()
    env.reset(seed=7)

    # Tower index 1 is the King tower.
    env.opponent_towers_hp[1] = 0

    _, reward, terminated, truncated, info = env.step(0)

    assert terminated is True
    assert truncated is False
    assert reward == 1.0
    assert info["winner"] == "player"


def test_same_seed_reproduces_same_match():
    first_env = CardBattleEnv()
    second_env = CardBattleEnv()

    first_env.reset(seed=7)
    second_env.reset(seed=7)

    # Both players wait; only the seeded random opponent makes choices.
    for _ in range(10):
        first_result = first_env.step(0)
        second_result = second_env.step(0)

        first_observation, first_reward, first_done, first_time, first_info = first_result
        second_observation, second_reward, second_done, second_time, second_info = second_result

        np.testing.assert_array_equal(
            first_observation,
            second_observation
        )

        assert first_reward == second_reward
        assert first_done == second_done
        assert first_time == second_time
        assert first_info == second_info