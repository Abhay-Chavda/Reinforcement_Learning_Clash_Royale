from ..base_card import Card


class Goblin(Card):
    def __init__(self):
        super().__init__(
            name="Goblin",
            elixir_cost=2,
            hp=250,
            damage=80,
            power=80,
            speed=0.40,
            attack_range=0.05,
            aggro_range=0.15,
            attack_cooldown=0.8,
            target_type="troops_and_buildings"
        )
