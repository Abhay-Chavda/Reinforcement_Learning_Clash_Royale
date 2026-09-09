from ..base_card import Card


class Archer(Card):
    def __init__(self):
        super().__init__(
            name="Archer",
            elixir_cost=3,
            hp=400,
            damage=150,
            power=150,
            speed=0.25,
            attack_range=0.20,
            aggro_range=0.20,
            attack_cooldown=1.2,
            target_type="troops_and_buildings"
        )
