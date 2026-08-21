from environment.card_library.cards.knight import Knight
from environment.card_library.cards.giant import Giant
from environment.combat import attack_unit
from environment.unit import Unit

from environment.targeting import find_target


# Create cards
knight_card = Knight()
giant_card = Giant()


# Create our units
our_knight = Unit(
    knight_card,
    [0.25, 0.30]
)

our_giant = Unit(
    giant_card,
    [0.70, 0.30]
)

enemy_knight = Unit(
    knight_card,
    [0.50, 0.95]
)
enemy_giant = Unit(
    giant_card,
    [0.25, 0.35]
)

enemy_units = [
    enemy_knight,
    enemy_giant
]


# Enemy towers
enemy_tower_positions = [
    [0.25, 0.90],
    [0.50, 0.90],
    [0.75, 0.90]
]

enemy_tower_hp = [2500, 5000, 2500]


# Test Knight
target_type, target = find_target(
    our_knight,
    enemy_units,
    enemy_tower_positions,
    enemy_tower_hp
)

print("Knight target:")
if target_type == "unit":
    print(target_type, target.name)

elif target_type == "tower":
    print(target_type, target)

else:
    print("No target")


# Test Giant
target_type, target = find_target(
    our_giant,
    enemy_units,
    enemy_tower_positions,
    enemy_tower_hp
)

print("\nGiant target:")
print(target_type, target)

print("Enemy Giant HP before:", enemy_giant.hp)

success = attack_unit(
    our_knight,
    enemy_giant
)

print("Attack successful:", success)
print("Enemy Giant HP after:", enemy_giant.hp)