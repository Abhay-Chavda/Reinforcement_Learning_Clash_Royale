import numpy as np

def get_distance(position_a,position_b):
    """Calculate Euclidian Distance between two positions."""

    position_a = np.array(position_a,dtype=np.float32)
    position_b = np.array(position_b,dtype=np.float32)

    return np.linalg.norm(position_a - position_b)

def find_nearest_enemy_troop(unit,enemy_units):

    """Find the Nearest alive enemy troop.
    Returns : The nearest enemy unit or None if no alive enemy units are found."""

    nearest_enemy = None
    nearest_distance = float('inf')
    aggro_range = unit.aggro_range

    for enemy_unit in enemy_units:

        #ignore dead units
        if not enemy_unit.alive:
            continue

        distance = get_distance(unit.position,enemy_unit.position)

        if distance < nearest_distance and distance <= aggro_range:
            nearest_distance = distance
            nearest_enemy = enemy_unit

    return nearest_enemy

def find_nearest_enemy_tower(unit,enemy_tower_positions,enemy_tower_hp):
    """Find the Nearest enemy tower.
    Returns : The nearest enemy tower position."""

    nearest_tower_index = None
    nearest_distance = float('inf')

    for tower_index, tower_position in enumerate(enemy_tower_positions):

        #ignore dead towers
        if enemy_tower_hp[tower_index] <= 0:
            continue

        distance = get_distance(
            unit.position,
            tower_position
        )

        if distance < nearest_distance:
            nearest_distance = distance
            nearest_tower_index = tower_index

    return nearest_tower_index

def find_target(
        unit,
        enemy_units,
        enemy_tower_positions,
        enemy_tower_hp
):
    """
    Select the target for a unit based on it's target type
    returns:
    ("Unit",enemy_unit)
    or
    ("tower",tower_index)
    or
    (None,None) if no target is found
    """

    if unit.target_type == "troops_and_buildings":

        enemy_unit = find_nearest_enemy_troop(
            unit,
            enemy_units
        )

        if enemy_unit is not None:
            return "unit",enemy_unit

        tower_index = find_nearest_enemy_tower(
            unit,
            enemy_tower_positions,
            enemy_tower_hp
        )

        if tower_index is not None:
            return "tower",tower_index
    
    #Troops that buildings only 

    elif unit.target_type == "buildings":

        tower_index = find_nearest_enemy_tower(
            unit,
            enemy_tower_positions,
            enemy_tower_hp
        )

        if tower_index is not None:
            return "tower",tower_index

    #not Valid target
    return None,None