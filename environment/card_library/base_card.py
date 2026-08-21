class Card:
    def __init__(
        self,
        name,
        elixir_cost,
        hp,
        damage,
        power,
        speed,
        attack_range,
        aggro_range,
        attack_cooldown,
        target_type
    ):
        self.name = name
        self.elixir_cost = elixir_cost
        self.hp = hp
        self.damage = damage
        self.power = power
        self.speed = speed
        self.attack_range = attack_range
        self.aggro_range = aggro_range
        self.attack_cooldown = attack_cooldown
        self.target_type = target_type


    def __repr__(self):
        return (
            f"{self.name}"
            f"HP={self.hp}, "
            f"Damage={self.damage}, "
            f"Power={self.power}, "
            f"Speed={self.speed}, "
            f"AttackRange={self.attack_range}, "
            f"AggroRange={self.aggro_range}, "
            f"AttackCooldown={self.attack_cooldown}"
            f"TargetType={self.target_type})"
        )

class Tower:
    def __init__(
            self,
            name,
            hp,
            damage,
            attack_cooldown,
            attack_range):
        self.name = name
        self.hp = hp
        self.damage = damage
        self.attack_cooldown = attack_cooldown
        self.attack_range = attack_range

    def __repr__(self):
        return(
            f"{self.name}"
            f"HP={self.hp}"
            f"Damage={self.damage}"
            f"AttackCooldown={self.attack_cooldown}"
            f"AttackRange={self.attack_range}"
        )