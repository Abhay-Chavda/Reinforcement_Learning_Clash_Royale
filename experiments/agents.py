"""Baseline agents that act from the 149-value observation.

Every agent exposes ``predict(obs, deterministic=True) -> (action, None)``,
the same signature as a Stable-Baselines3 model, so it can be used either as
the learning side's policy during evaluation or as ``opponent_policy`` inside
``CardBattleEnv`` (which hands it the mirrored, opponent-perspective view).
"""

import numpy as np

from environment.card_battle_env import CardBattleEnv

KNIGHT_COST = 3
GIANT_COST = 5
MAX_ELIXIR = 10.0
UNIT_SLOTS, UNIT_FEATURES = 10, 7


def _decode(obs):
    obs = np.asarray(obs, dtype=np.float32)
    own_elixir = obs[6] * MAX_ELIXIR
    enemy_queen_hp = {"left": obs[3], "right": obs[5]}
    enemy_units = obs[9 + UNIT_SLOTS * UNIT_FEATURES:].reshape(UNIT_SLOTS, UNIT_FEATURES)
    return own_elixir, enemy_queen_hp, enemy_units


class RandomAgent:
    """Uniformly random among currently affordable actions (incl. wait)."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def predict(self, obs, deterministic=True):
        elixir, _, _ = _decode(obs)
        actions = [CardBattleEnv.WAIT]
        if elixir >= KNIGHT_COST:
            actions += [CardBattleEnv.KNIGHT_LEFT, CardBattleEnv.KNIGHT_RIGHT]
        if elixir >= GIANT_COST:
            actions += [CardBattleEnv.GIANT_LEFT, CardBattleEnv.GIANT_RIGHT]
        return int(self.rng.choice(actions)), None


class HeuristicAgent:
    """Hand-written 'defend, then push' rule policy.

    1. If an enemy troop has crossed into our half of a lane, defend that lane
       with a Knight (when affordable).
    2. Otherwise save elixir; at >= 8 elixir launch a Giant in the lane whose
       enemy Queen tower is weaker (ties go left), keeping 3 for a defender.
    3. Otherwise wait.
    """

    def __init__(self, push_threshold=8.0):
        self.push_threshold = push_threshold

    def predict(self, obs, deterministic=True):
        elixir, enemy_queen_hp, enemy_units = _decode(obs)
        active = enemy_units[enemy_units[:, 0] > 0.5]
        for lane_value, action in ((0.0, CardBattleEnv.KNIGHT_LEFT), (1.0, CardBattleEnv.KNIGHT_RIGHT)):
            threats = active[(active[:, 2] == lane_value) & (active[:, 4] < 0.5)]
            if len(threats) and elixir >= KNIGHT_COST:
                return action, None
        if elixir >= self.push_threshold:
            lane = "left" if enemy_queen_hp["left"] <= enemy_queen_hp["right"] else "right"
            return (CardBattleEnv.GIANT_LEFT if lane == "left" else CardBattleEnv.GIANT_RIGHT), None
        return CardBattleEnv.WAIT, None
