from ..base_card import Tower


class PrincessTower(Tower):
    def __init__(self):
        super().__init__(
            name = "princess",
            hp = 2000,
            damage = 100,
            attack_range = 0.20,
            attack_cooldown = 1
            )