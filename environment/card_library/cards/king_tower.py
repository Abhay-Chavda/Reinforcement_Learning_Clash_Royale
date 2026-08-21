from ..base_card import Tower


class PrincessTower(Tower):
    def __init__(self):
        super().__init__(
            name = "King",
            hp = 3000,
            damage = 150,
            attack_range = 0.25,
            attack_cooldown = 1
            )