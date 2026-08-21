import numpy as np

class Unit:

    def __init__(self,troop_type,position):
        self.troop_type = troop_type

        self.name = troop_type.name
        self.elixir_cost = troop_type.elixir_cost
        self.max_hp = troop_type.hp
        self.hp = troop_type.hp
        self.damage = troop_type.damage
        self.speed = troop_type.speed
        self.attack_range = troop_type.attack_range
        self.aggro_range = troop_type.aggro_range
        self.target_type = troop_type.target_type

        #position
        self.position = np.array(position,dtype=np.float32)

        #state
        self.alive = True

    def move_towards(self, target_position,dt):
        '''Move the unit towards the target position based on its speed and the time step dt.'''

        target_position = np.array(
            target_position,
            dtype=np.float32
            )

        direction = target_position - self.position

        distance = np.linalg.norm(direction)

        #already at the target position
        if distance == 0:
            return

        movement = min(self.speed * dt, distance)
        direction = direction / distance  
        self.position += direction * movement

    def take_damage(self, amount):
        self.hp -= amount

        if self.hp <= 0:
            self.alive = False
            self.hp = 0
        