from ..base_card import Card


class Knight(Card):
    def __init__(self):
        super().__init__(
            name="Knight",
            elixir_cost = 3,
            hp=1000,
            damage=100,
            power=100,
            speed=0.20,
            attack_range=0.05,
            aggro_range=0.20,
            attack_cooldown=1.0,
            target_type="troops_and_buildings"
        )
