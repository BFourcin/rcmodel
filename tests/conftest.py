import pytest
import torch
from rcmodel import Room, Building, InputScaling, RCModel


@pytest.fixture
def rooms_n2():
    """
    Two rooms side by side
    """
    rm1 = Room("rm1", [[0, 0], [5, 0], [5, 5], [0, 5]])
    rm2 = Room("rm2", [[5, 0], [10, 0], [10, 5], [5, 5]])
    rooms = [rm1, rm2]

    return rooms


@pytest.fixture
def rooms_n9():
    """
    Complicated arrangement of rooms
    """
    rooms = []

    rooms.append(Room("rm1", [[0, 0], [5, 0], [5, 5], [0, 5]]))
    rooms.append(Room("rm2", [[5, 0], [10, 0], [10, 2], [10, 4], [10, 5], [5, 5]]))
    rooms.append(Room("rm3", [[0, 5], [5, 5], [10, 5], [10, 6], [10, 8], [10, 10]]))

    rooms.append(Room("rm4", [[10, 10], [12, 10], [12, 8], [10, 8]]))
    rooms.append(Room("rm5", [[10, 8], [12, 8], [12, 6], [10, 6]]))
    rooms.append(Room("rm6", [[10, 6], [12, 6], [12, 4], [10, 4], [10, 5]]))
    rooms.append(Room("rm7", [[10, 4], [12, 4], [12, 2], [10, 2]]))
    rooms.append(Room("rm8", [[10, 2], [12, 2], [12, 0], [10, 0]]))
    rooms.append(
        Room(
            "rm9",
            [[12, 0], [12, 2], [12, 4], [12, 6], [12, 8], [12, 10], [14, 10], [14, 0]],
        )
    )

    return rooms


def get_building(rooms):
    height = 1
    rm_cap = [300]
    Ce = [1e3, 8e2]
    Re = [5, 1, 0.5]
    Rint = [0.1]

    theta = [rm_cap, Ce, Re, Rint]
    theta = [item for sublist in theta for item in sublist]  # flatten list

    bld = Building(rooms, height)
    bld.update_inputs(theta)

    return bld


@pytest.fixture
def building_n2(rooms_n2):
    return get_building(rooms_n2)


@pytest.fixture
def building_n9(rooms_n9):
    return get_building(rooms_n9)


def get_model(building):
    # Function for constant outside temperature. try/except allows for broadcasting of value.
    def dummy_tout(t):
        try:
            return -5 * torch.ones(len(t))

        except TypeError:
            return torch.tensor(-5)

    rm_CA = [200, 800]  # [min, max] Capacitance/m2
    C1 = [1.5 * 10**4, 10**6]  # Capacitance
    C2 = [1.5 * 10**4, 10**6]
    R1 = [0.2, 1.2]  # Resistance ((K.m^2)/W)
    R2 = [0.2, 1.2]
    R3 = [0.2, 1.2]
    Rin = [0.2, 1.2]
    cool = [0, 5000]  # Cooling limit in W/m2
    gain = [0, 5000]  # gain limit (W/m^2) no additional energy

    scaling = InputScaling(rm_CA, C1, C2, R1, R2, R3, Rin, cool, gain)

    # Initialise RCModel with the building and InputScaling
    transform = torch.sigmoid
    model = RCModel(building, scaling, dummy_tout, transform)
    return model


@pytest.fixture
def model_n2(building_n2):
    return get_model(building_n2)


@pytest.fixture
def model_n9(building_n9):
    return get_model(building_n9)
