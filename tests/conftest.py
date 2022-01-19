import pytest

from rcmodel.building import Building
from rcmodel.room import Room


@pytest.fixture
def rooms_n2():
    """
    Two rooms side by side
    """
    rm1 = Room('rm1', [[0,0], [5,0], [5,5], [0,5]])
    rm2 = Room('rm2', [[5,0], [10,0], [10,5], [5,5]])
    rooms = [rm1, rm2]

    return rooms


@pytest.fixture
def rooms_n9():
    """
    Complicated arrangement of rooms
    """
    rooms = []

    rooms.append(Room('rm1', [[0,0], [5,0], [5,5], [0,5]]))
    rooms.append(Room('rm2', [[5,0], [10,0], [10,2], [10,4], [10,5], [5,5]]))
    rooms.append(Room('rm3', [[0,5], [5,5], [10,5], [10,6], [10,8], [10,10]]))

    rooms.append(Room('rm4', [[10,10], [12,10], [12,8], [10,8]]))
    rooms.append(Room('rm5', [[10,8], [12,8], [12,6], [10,6]]))
    rooms.append(Room('rm6', [[10,6], [12,6], [12,4], [10,4], [10,5]]))
    rooms.append(Room('rm7', [[10,4], [12,4], [12,2], [10,2]]))
    rooms.append(Room('rm8', [[10,2], [12,2], [12,0], [10,0]]))
    rooms.append(Room('rm9', [[12,0], [12,2], [12,4], [12,6], [12,8], [12,10], [14,10], [14,0]]))

    return rooms


@pytest.fixture
def building_n2(rooms_n2):
    height = 1
    rm_cap = [300]
    Ce = [1e3, 8e2]
    Re = [5, 1, 0.5]
    Rint = [0.1]
    gain = [0]

    theta = [rm_cap, Ce, Re, Rint, gain]
    theta = [item for sublist in theta for item in sublist]  # flatten list

    bld = Building(rooms_n2, height)
    bld.update_inputs(theta)

    return bld


@pytest.fixture
def building_n9(rooms_n9):
    height = 1
    rm_cap = [300]
    Ce = [1e3, 8e2]
    Re = [5, 1, 0.5]
    Rint = [0.1]
    gain = [0]

    theta = [rm_cap, Ce, Re, Rint, gain]
    theta = [item for sublist in theta for item in sublist]  # flatten list

    bld = Building(rooms_n9, height)
    bld.update_inputs(theta)

    return bld

