import pytest

from rcmodel.building import Building
from rcmodel.room import Room


@pytest.fixture
def rooms_n2():
    """
    Two rooms side by side
    """
    rm1 = Room('rm1', 1e3, [[0,0], [5,0], [5,5], [0,5]])
    rm2 = Room('rm2', 2e3, [[5,0], [10,0], [10,5], [5,5]])
    rooms = [rm1, rm2]

    return rooms


@pytest.fixture
def rooms_n9():
    """
    Complicated arrangement of rooms
    """
    rooms = []

    rooms.append(Room('rm1', 1e3, [[0,0], [5,0], [5,5], [0,5]]))
    rooms.append(Room('rm2', 2e3, [[5,0], [10,0], [10,2], [10,4], [10,5], [5,5]]))
    rooms.append(Room('rm3', 3e3, [[0,5], [5,5], [10,5], [10,6], [10,8], [10,10]]))

    rooms.append(Room('rm4', 4e3, [[10,10], [12,10], [12,8], [10,8]]))
    rooms.append(Room('rm5', 5e3, [[10,8], [12,8], [12,6], [10,6]]))
    rooms.append(Room('rm6', 6e3, [[10,6], [12,6], [12,4], [10,4], [10,5]]))
    rooms.append(Room('rm7', 7e3, [[10,4], [12,4], [12,2], [10,2]]))
    rooms.append(Room('rm8', 8e3, [[10,2], [12,2], [12,0], [10,0]]))
    rooms.append(Room('rm9', 9e3, [[12,0], [12,2], [12,4], [12,6], [12,8], [12,10], [14,10], [14,0]]))

    return rooms


@pytest.fixture
def building_n2(rooms_n2):
    height = 1
    Re = [5, 1, 0.5]
    Ce = [1e3,8e2]
    Rint = 0.1

    bld = Building(rooms_n2, height, Re, Ce, Rint)

    return bld


@pytest.fixture
def building_n9(rooms_n9):
    height = 1
    Re = [5, 1, 0.5]
    Ce = [1e3, 8e2]
    Rint = 0.1

    bld = Building(rooms_n9, height, Re, Ce, Rint)

    return bld
