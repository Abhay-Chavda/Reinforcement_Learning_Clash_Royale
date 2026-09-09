import numpy as np 

def get_distance(position_a,position_b):
    """Calculate Euclidian Distance between two positions."""

    position_a = np.array(position_a,dtype=np.float32)
    position_b = np.array(position_b,dtype=np.float32)

    return np.linalg.norm(position_a - position_b)

def is_in_attack_range(attacker,target_position):
    """Check Whether the target is within the attack range of the attacker."""

    distance = get_distance(
        attacker.position,
        target_position)

    return distance <= attacker.attack_range

def attack_unit(attacker,target):
    """Attack the target if within attacker's range or alive"""

    if not attacker.alive:
        print("Attacker is dead")
        return False

    if not target.alive:
        return False

    if not attacker.can_attack():
        return False

    if not is_in_attack_range(
        attacker,
        target.position):
        print("Target is out of range")
        return False

    target.take_damage(attacker.damage)
    attacker.reset_attack_cooldown()

    return True


def attack_tower(
        attacker,
        tower_index,
        tower_positions,
        tower_hp):
    """Attack the target tower if within attacker's range or alive"""

    if not attacker.alive:
        return False

    if tower_index < 0 or tower_index >= len(tower_positions):
        return False

    if tower_hp[tower_index] <= 0:
        return False

    if not attacker.can_attack():
        return False

    if not is_in_attack_range(
        attacker,
        tower_positions[tower_index]):
        return False

    tower_hp[tower_index] -= attacker.damage

    if tower_hp[tower_index] < 0:
        tower_hp[tower_index] = 0

    attacker.reset_attack_cooldown()

    return True
