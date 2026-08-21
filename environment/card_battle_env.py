import gymnasium as gym
import numpy as np
from environment.unit import Unit
from environment.card_library.cards import Knight, Giant,Goblin,Archer
from environment.targeting import find_target
from environment.combat import attack_unit, attack_tower

class CardBattleEnv(gym.Env):

    def __init__(self):
        super().__init__()

        #area dimensions
        self.arena_width = 1.0
        self.arena_height = 1.0

        #towers position
        self.player_towers_position = np.array([[0.25,0.22],[0.5,0.1],[0.75,0.22]])
        self.opponent_towers_position = np.array([[0.25,0.9],[0.5,0.9],[0.75,0.9]])

        #towers health
        self.max_tower_hp = np.array([2500,5000,2500])

        self.player_towers_hp = self.max_tower_hp.copy()
        self.opponent_towers_hp = self.max_tower_hp.copy()
        self.player_units = []
        self.opponent_units = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        #reset towers health
        self.player_towers_hp = self.max_tower_hp.copy()
        self.opponent_towers_hp = self.max_tower_hp.copy()

        #remove all units from the arena
        self.player_units = []
        self.opponent_units = []

        #spawn one test unit for each player    
        player_card = Giant()
        player_unit = Unit(
            troop_type =  player_card,
            position = [0.5, 0.2]
        )

        self.player_units.append(player_unit)

        # opponent_card = None
        # opponent_unit = Unit(
        #     troop_type = opponent_card,
        #     position = [0.5, 0.8]
        # )
        # self.opponent_units.append(opponent_unit)

        observation = np.concatenate(
            [
                self.player_towers_hp,
                self.opponent_towers_hp
            ]
        ).astype(np.float32)

        info = {}

        return observation, info

    def render(self):
        print("\n=========Arena==========")

        print("Player Towers")
        for i,hp in enumerate(self.player_towers_hp):
            print(f"Tower {i}: HP = {hp}")

        print("\nOpponent Towers")
        for i,hp in enumerate(self.opponent_towers_hp):
            print(f"Tower {i}: HP = {hp}")

        print("\n=========================")

    def update_units(self, dt):
        #Player Units
        for unit in self.player_units:
            if not unit.alive:
                continue

            target_type, target = find_target(
                unit,
                self.opponent_units,
                self.opponent_towers_position,
                self.opponent_towers_hp
            )

            #no target found
            if target_type is None:
                continue

            print(
                f"{unit.name} -> {target_type}"
                f"{target.name if target_type=='unit' else target}"
            )
            #enemy troop target

            if target_type == "unit":

                target_position = target.position

                distance = np.linalg.norm(target_position - unit.position)
                print(
                    f"{unit.name} -> tower{target} | "
                    f"distance={distance:.4f} | "
                    f"attack_range={unit.attack_range}"
                )


                if distance > unit.attack_range:
                    unit.move_towards(target_position, dt)

                else: 
                    attack_unit(unit, target)

            #enemy tower target

            elif target_type == "tower":

                target_position = self.opponent_towers_position[target]

                distance = np.linalg.norm(target_position - unit.position)

                print(
                    f"{unit.name} -> tower{target} | "
                    f"distance={distance:.4f} | "
                    f"attack_range={unit.attack_range}"
                )

                if distance > unit.attack_range:
                    unit.move_towards(target_position, dt)
                else:
                    attack_tower(
                        unit,
                        target,
                        self.opponent_towers_position,
                        self.opponent_towers_hp
                    )

        #Enemy Units

        for unit in self.opponent_units:

            if not unit.alive:
                continue

            target_type, target = find_target(
                unit,
                self.player_units,
                self.player_towers_position,
                self.player_towers_hp
            )
            #no target found
            if target_type is None:
                continue

            #player troop target

            if target_type == "unit":

                target_position = target.position
                distance = np.linalg.norm(target_position - unit.position)

                if distance > unit.attack_range:
                    unit.move_towards(target_position, dt)

                else:
                    attack_unit(unit, target)

                
            #player tower target

            elif target_type == "tower":
                target_position = self.player_towers_position[target]
                distance = np.linalg.norm(target_position - unit.position)

                if distance > unit.attack_range:
                    unit.move_towards(target_position, dt)

                else:
                    attack_tower(
                        unit,
                        target,
                        self.player_towers_position,
                        self.player_towers_hp
                    )

    def step(self, action):

        dt = 0.1  #time step

        self.update_units(dt)

        observation = np.concatenate(
            [
                self.player_towers_hp,
                self.opponent_towers_hp
            ]
        ).astype(np.float32)

        reward = 0.0

        terminated = False
        truncated = False

        info = {}

        return(
            observation,
            reward,
            terminated,
            truncated,
            info
        )