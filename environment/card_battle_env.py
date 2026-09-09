"""Small deterministic two-lane environment for the first RL milestone.

Action mapping: 0 wait, 1 Knight-left, 2 Knight-right,
3 Giant-left, 4 Giant-right.
"""

import gymnasium as gym
import numpy as np

from environment.card_library.cards import Giant, Knight
from environment.combat import attack_tower, attack_unit, get_distance
from environment.unit import Unit


class CardBattleEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    WAIT, KNIGHT_LEFT, KNIGHT_RIGHT, GIANT_LEFT, GIANT_RIGHT = range(5)
    LANES = ("left", "right")
    LANE_X = {"left": 0.25, "right": 0.75}

    def __init__(self, game_duration=180.0, dt=0.5, reward_shaping=False):
        super().__init__()
        self.dt = dt
        self.game_duration = game_duration
        self.max_elixir = 10.0
        self.elixir_regen_per_second = 0.5
        self.max_tower_hp = 2500.0
        self.tower_damage = 90.0
        self.tower_attack_range = 0.20
        self.tower_attack_cooldown = 1.0
        self.cards = {"Knight": Knight, "Giant": Giant}

        self.action_space = gym.spaces.Discrete(5)
        # player tower HP (2), opponent tower HP (2), both elixirs, time left
        # 9 basic features + (10 player slots + 10 opponent slots) × 7 features
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(149,),
            dtype=np.float32
        )
        self.max_units_per_side = 10
        self.unit_feature_count = 7
        self.reset()
        self.reward_shaping = reward_shaping
        self._prev_player_towers_hp = None
        self._prev_opponent_towers_hp = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_towers_hp = np.array(
            [2500.0, 5000.0, 2500.0],
            
            dtype=np.float32
        )

        self.opponent_towers_hp = np.array(
            [2500.0, 5000.0, 2500.0],
            dtype=np.float32
        )

        # Order: left Queen, King, right Queen
        self.player_tower_positions = np.array([
            [0.25, 0.08],
            [0.50, 0.08],
            [0.75, 0.08],
        ], dtype=np.float32)

        self.opponent_tower_positions = np.array([
            [0.25, 0.92],
            [0.50, 0.92],
            [0.75, 0.92],
        ], dtype=np.float32)
        self.player_units, self.opponent_units = [], []
        self.player_elixir = self.opponent_elixir = 5.0
        self._prev_player_tower_hp = self.player_towers_hp.copy()
        self._prev_opponent_tower_hp = self.opponent_towers_hp.copy()
        self.time_remaining = self.game_duration
        self.opponent_action_timer = 0.0
        self.player_tower_timers = np.zeros(3, dtype=np.float32)
        self.opponent_tower_timers = np.zeros(3, dtype=np.float32)
        return self._observation(), {}

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError("Action must be an integer from 0 to 4.")

        invalid_action = not self._deploy_from_action(action, "player")
        self._regenerate_elixir()
        self._maybe_deploy_opponent()
        self._update_battlefield()
        self.time_remaining = max(0.0, self.time_remaining - self.dt)
    
        terminated, winner = self._terminal_state()
        truncated = self.time_remaining <= 0.0 and not terminated
        if truncated:
            winner = self._winner_on_time()
        reward = 1.0 if winner == "player" else -1.0 if winner == "opponent" else 0.0
        if self.reward_shaping:
            opponent_damage = max(0.0, self._prev_opponent_tower_hp.sum() - self.opponent_towers_hp.sum())
            player_damage = max(0.0, self._prev_player_tower_hp.sum() - self.player_towers_hp.sum())
            reward += 0.0002 * (opponent_damage - player_damage)
            if invalid_action:
                reward -= 0.01
        self._prev_player_tower_hp = self.player_towers_hp.copy()
        self._prev_opponent_tower_hp = self.opponent_towers_hp.copy()
        info = {"winner": winner, "invalid_action": invalid_action}
        return self._observation(), reward, terminated, truncated, info

    def _deploy_from_action(self, action, team):
        if action == self.WAIT:
            return True
        card_name, lane = {
            self.KNIGHT_LEFT: ("Knight", "left"),
            self.KNIGHT_RIGHT: ("Knight", "right"),
            self.GIANT_LEFT: ("Giant", "left"),
            self.GIANT_RIGHT: ("Giant", "right"),
        }[action]
        card = self.cards[card_name]()
        elixir_name = f"{team}_elixir"
        if getattr(self, elixir_name) < card.elixir_cost:
            return False
        spawn_y = 0.15 if team == "player" else 0.85
        unit = Unit(card, [self.LANE_X[lane], spawn_y], lane=lane, team=team)
        getattr(self, f"{team}_units").append(unit)
        setattr(self, elixir_name, getattr(self, elixir_name) - card.elixir_cost)
        return True

    def _regenerate_elixir(self):
        amount = self.elixir_regen_per_second * self.dt
        self.player_elixir = min(self.max_elixir, self.player_elixir + amount)
        self.opponent_elixir = min(self.max_elixir, self.opponent_elixir + amount)

    def get_target_tower(self, lane, enemy_towers_hp):

        if lane == "left":
            queen_index = 0
        else:
            queen_index = 2

        king_index = 1

        if enemy_towers_hp[queen_index] > 0:
            return queen_index

        if enemy_towers_hp[king_index] > 0:
            return king_index

        return None

    def _maybe_deploy_opponent(self):
        self.opponent_action_timer -= self.dt
        if self.opponent_action_timer > 0.0:
            return
        self.opponent_action_timer = 1.0
        affordable_actions = [self.WAIT]
        for action in range(1, 5):
            card_name = "Knight" if action <= 2 else "Giant"
            if self.cards[card_name]().elixir_cost <= self.opponent_elixir:
                affordable_actions.append(action)
        self._deploy_from_action(int(self.np_random.choice(affordable_actions)), "opponent")

    def _update_battlefield(self):
        for unit in self.player_units + self.opponent_units:
            unit.update_cooldown(self.dt)
        self._update_units("player")
        self._update_units("opponent")
        self._update_towers("player")
        self._update_towers("opponent")
        self.player_units = [unit for unit in self.player_units if unit.alive]
        self.opponent_units = [unit for unit in self.opponent_units if unit.alive]

    def _update_units(self, team):
        units = getattr(self, f"{team}_units")
        enemy_team = "opponent" if team == "player" else "player"
        enemies = getattr(self, f"{enemy_team}_units")
        enemy_towers = self.opponent_towers_hp if team == "player" else self.player_towers_hp
        tower_y, direction = (0.92, 1.0) if team == "player" else (0.08, -1.0)
        for unit in units:
            if not unit.alive:
                continue
            targets = [enemy for enemy in enemies if enemy.alive and enemy.lane == unit.lane]
            target = min(targets, key=lambda x: get_distance(unit.position, x.position), default=None)
            if target is not None:
                if get_distance(unit.position, target.position) <= unit.attack_range:
                    attack_unit(unit, target)
                else:
                    unit.move_towards(target.position, self.dt)
                continue
            # lane_index = self.LANES.index(unit.lane)
            # tower_position = np.array([self.LANE_X[unit.lane], tower_y], dtype=np.float32)
            # if get_distance(unit.position, tower_position) <= unit.attack_range:
            #     # A one-element view lets existing attack_tower update the real HP array.
            #     attack_tower(unit, 0, [tower_position], enemy_towers[lane_index:lane_index + 1])
            # else:
            #     unit.position[1] = np.clip(unit.position[1] + direction * unit.speed * self.dt, 0.0, 1.0)
            enemy_tower_positions = (
            self.opponent_tower_positions
            if team == "player"
            else self.player_tower_positions
            )

            tower_index = self.get_target_tower(
                unit.lane,
                enemy_towers
            )

            if tower_index is None:
                continue

            tower_position = enemy_tower_positions[tower_index]

            if get_distance(unit.position, tower_position) <= unit.attack_range:
                attack_tower(
                    unit,
                    tower_index,
                    enemy_tower_positions,
                    enemy_towers
                )
            else:
                unit.move_towards(tower_position, self.dt)

    def _update_towers(self, team):

        timers = (
            self.player_tower_timers
            if team == "player"
            else self.opponent_tower_timers
        )

        tower_hp = (
            self.player_towers_hp
            if team == "player"
            else self.opponent_towers_hp
        )

        tower_positions = (
            self.player_tower_positions
            if team == "player"
            else self.opponent_tower_positions
        )

        enemies = (
            self.opponent_units
            if team == "player"
            else self.player_units
        )

        for tower_index, tower_position in enumerate(tower_positions):

            # Reduce this tower's cooldown timer.
            timers[tower_index] = max(
                0.0,
                timers[tower_index] - self.dt
            )

            # Destroyed tower or still on cooldown: cannot attack.
            if tower_hp[tower_index] <= 0 or timers[tower_index] > 0:
                continue

            # King tower uses a slightly larger defensive range.
            if tower_index == 1:
                attack_range = 0.30
                possible_targets = [
                    unit for unit in enemies
                    if unit.alive
                    and get_distance(unit.position, tower_position) <= attack_range
                ]

            # Queen towers defend only their own lane.
            else:
                attack_range = self.tower_attack_range

                possible_targets = [
                    unit for unit in enemies
                    if unit.alive
                    and get_distance(unit.position, tower_position) <= attack_range
                ]   

            if possible_targets:
                target = min(
                    possible_targets,
                    key=lambda unit: get_distance(
                        unit.position,
                        tower_position
                    )
                )

                target.take_damage(self.tower_damage)

                # Tower must wait before its next attack.
                timers[tower_index] = self.tower_attack_cooldown

    def _terminal_state(self):
        if np.any(self.opponent_towers_hp[1] <= 0):
            return True, "player"
        if np.any(self.player_towers_hp[1] <= 0):
            return True, "opponent"
        return False, None

    def _winner_on_time(self):
        player_hp, opponent_hp = self.player_towers_hp.sum(), self.opponent_towers_hp.sum()
        if player_hp == opponent_hp:
            return "draw"
        return "player" if player_hp > opponent_hp else "opponent"


    def _encode_units(self, units):
        """
        Convert a variable-length troop list into 10 fixed observation slots.

        Each unit has 7 values:
        active, card type, lane, x position, y position,
        normalized HP, normalized cooldown.
        """

        encoded = np.zeros(
            (self.max_units_per_side, self.unit_feature_count),
            dtype=np.float32
        )

        for index, unit in enumerate(units[:self.max_units_per_side]):

            card_type = 0.5 if unit.name == "Knight" else 1.0
            lane = 0.0 if unit.lane == "left" else 1.0

            encoded[index] = [
                1.0,  # This troop slot is active.
                card_type,
                lane,
                unit.position[0],
                unit.position[1],
                unit.hp / unit.max_hp,
                unit.time_until_next_attack / unit.attack_cooldown,
            ]

        return encoded.flatten()
    
    def _observation(self):
        tower_max_hp = np.array(
            [2500.0, 5000.0, 2500.0],
            dtype=np.float32
        )

        basic_state = np.array([
            *self.player_towers_hp / tower_max_hp,
            *self.opponent_towers_hp / tower_max_hp,
            self.player_elixir / self.max_elixir,
            self.opponent_elixir / self.max_elixir,
            self.time_remaining / self.game_duration,
        ], dtype=np.float32)

        player_units_state = self._encode_units(
            self.player_units
        )

        opponent_units_state = self._encode_units(
            self.opponent_units
        )

        return np.concatenate([
            basic_state,
            player_units_state,
            opponent_units_state,
        ]).astype(np.float32)

    def render(self):
        print(f"time={self.time_remaining:.1f}; elixir={self.player_elixir:.1f}/{self.opponent_elixir:.1f}")
        print(f"towers player={self.player_towers_hp}; opponent={self.opponent_towers_hp}")
        print(f"active units player={len(self.player_units)}; opponent={len(self.opponent_units)}")
