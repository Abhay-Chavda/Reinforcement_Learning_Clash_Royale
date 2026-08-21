from ..base_card import Card


class Giant(Card):
    def __init__(self):
        super().__init__(
            name="Giant",
            elixir_cost = 5,
            hp=2000,
            damage=200,
            power=200,
            speed=0.10,
            attack_range=0.05,
            aggro_range=0.25,
            target_type="buildings"
        )