import pytest
import torch
import numpy as np

from rcmodel.building import Building


def rounded(x):
    """torch doesent round to n decimal place so this function does that"""
    n_digits = 4
    return torch.round(x * 10 ** n_digits) / (10 ** n_digits)


# -------- Tests --------

def test_rounded():
    x = torch.tensor([[0.71937193, 0.87787193, 0.40067193],
                      [0.90667193, 0.12417193, 0.32707193],
                      [0.71237193, 0.27967193, 0.01457193]])
    y = torch.tensor([[0.7194, 0.8779, 0.4007],
                      [0.9067, 0.1242, 0.3271],
                      [0.7124, 0.2797, 0.0146]])
    assert torch.equal(rounded(x), y), "function not part of the model but is used for testing"


@pytest.mark.parametrize(
    'building, num_walls',
    [
        (pytest.lazy_fixture('building_n2'), 7),
        (pytest.lazy_fixture('building_n9'), 27)
    ])
def test_unique_walls(building, num_walls):
    assert len(building.Walls) == num_walls, "Num walls counted incorrectly"


@pytest.mark.parametrize(
    'building, external_area',
    [
        (pytest.lazy_fixture('building_n2'), 30),
        (pytest.lazy_fixture('building_n9'), 44.180)
    ])
def test_surf_area(building, external_area):
    assert round(building.surf_area.item(), 3) == external_area


@pytest.mark.parametrize(
    'building, K',
    [
        (pytest.lazy_fixture('building_n2'), torch.tensor([[0., 30., 30.],
                                                           [30., 0., 50.],
                                                           [30., 50., 0.]])),
        (pytest.lazy_fixture('building_n9'),
         torch.tensor([[0.0000, 20.0000, 10.0000, 22.3607, 4.0000, 0.0000, 0.0000, 0.0000, 4.0000, 28.0000],
                       [20.0000, 0.0000, 50.0000, 50.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                       [10.0000, 50.0000, 0.0000, 50.0000, 0.0000, 0.0000, 10.0000, 20.0000, 20.0000, 0.0000],
                       [22.3607, 50.0000, 50.0000, 0.0000, 20.0000, 20.0000, 10.0000, 0.0000, 0.0000, 0.0000],
                       [4.0000, 0.0000, 0.0000, 20.0000, 0.0000, 20.0000, 0.0000, 0.0000, 0.0000, 20.0000],
                       [0.0000, 0.0000, 0.0000, 20.0000, 20.0000, 0.0000, 20.0000, 0.0000, 0.0000, 20.0000],
                       [0.0000, 0.0000, 10.0000, 10.0000, 0.0000, 20.0000, 0.0000, 20.0000, 0.0000, 20.0000],
                       [0.0000, 0.0000, 20.0000, 0.0000, 0.0000, 0.0000, 20.0000, 0.0000, 20.0000, 20.0000],
                       [4.0000, 0.0000, 20.0000, 0.0000, 0.0000, 0.0000, 0.0000, 20.0000, 0.0000, 20.0000],
                       [28.0000, 0.0000, 0.0000, 0.0000, 20.0000, 20.0000, 20.0000, 20.0000, 20.0000, 0.0000]])),
    ])
def test_connectivity_matrix(building, K):
    Re = [5, 1, 0.5]
    Ce = [1e3,8e2]
    Rint = 0.1

    k = building.make_connectivity_matrix()

    assert torch.equal(rounded(K), rounded(k))


def test_make_system_matrix_rooms_n2(building_n2):
    Re = building_n2.Re
    Ce = building_n2.Ce
    Rint = building_n2.Rint
    Crm1 = building_n2.rooms[0].capacitance
    Crm2 = building_n2.rooms[1].capacitance

    A = [[-30 / (Re[0] * Ce[0]) - 30 / (Re[1] * Ce[0]), 30 / (Re[1] * Ce[0]), 0, 0],
         [30 / (Re[1] * Ce[1]), -30 / (Re[1] * Ce[1]) - 30 / (Re[2] * Ce[1]), 15 / (Re[2] * Ce[1]),
          15 / (Re[2] * Ce[1])],
         [0, 15 / (Re[2] * Crm1), -15 / (Re[2] * Crm1) - 5 / (Rint * Crm1), 5 / (Rint * Crm1)],
         [0, 15 / (Re[2] * Crm2), 5 / (Rint * Crm2), -15 / (Re[2] * Crm2) - 5 / (Rint * Crm2)]]

    A = torch.tensor(A, dtype=torch.float32)

    assert torch.equal(rounded(A), rounded(building_n2.make_system_matrix()))


def test_make_system_matrix_rooms_n9(building_n9):
    Re = building_n9.Re
    Ce = building_n9.Ce
    Rint = building_n9.Rint

    Crm = [room.capacitance for room in building_n9.rooms]

    sa = building_n9.surf_area.item()  # Need .item() otherwise error

    A = [[-sa / (Re[0] * Ce[0]) - sa / (Re[1] * Ce[0]), sa / (Re[1] * Ce[0]), 0, 0, 0, 0, 0, 0, 0, 0, 0],  # ex1
         [sa / (Re[1] * Ce[1]), -sa / (Re[1] * Ce[1]) - sa / (Re[2] * Ce[1]), 10 / (Re[2] * Ce[1]), 5 / (Re[2] * Ce[1]),
          np.sqrt(10 ** 2 + 5 ** 2) / (Re[2] * Ce[1]), 2 / (Re[2] * Ce[1]), 0, 0, 0, 2 / (Re[2] * Ce[1]),
          14 / (Re[2] * Ce[1])],  # ex2
         [0, 10 / (Re[2] * Crm[0]), -10 / (Re[2] * Crm[0]) - 10 / (Rint * Crm[0]), 5 / (Rint * Crm[0]), 5 / (Rint * Crm[0]), 0, 0, 0,
          0, 0, 0],  # rm1
         [0, 5 / (Re[2] * Crm[1]), 5 / (Rint * Crm[1]), -5 / (Re[2] * Crm[1]) - 15 / (Rint * Crm[1]), 5 / (Rint * Crm[1]), 0, 0,
          1 / (Rint * Crm[1]), 2 / (Rint * Crm[1]), 2 / (Rint * Crm[1]), 0],  # rm2
         [0, np.sqrt(10 ** 2 + 5 ** 2) / (Re[2] * Crm[2]), 5 / (Rint * Crm[2]), 5 / (Rint * Crm[2]),
          -np.sqrt(10 ** 2 + 5 ** 2) / (Re[2] * Crm[2]) - 15 / (Rint * Crm[2]), 2 / (Rint * Crm[2]), 2 / (Rint * Crm[2]),
          1 / (Rint * Crm[2]), 0, 0, 0],  # rm3
         [0, 2 / (Re[2] * Crm[3]), 0, 0, 2 / (Rint * Crm[3]), -2 / (Re[2] * Crm[3]) - 6 / (Rint * Crm[3]), 2 / (Rint * Crm[3]), 0, 0,
          0, 2 / (Rint * Crm[3])],  # rm4
         [0, 0, 0, 0, 2 / (Rint * Crm[4]), 2 / (Rint * Crm[4]), -8 / (Rint * Crm[4]), 2 / (Rint * Crm[4]), 0, 0, 2 / (Rint * Crm[4])],
         # rm5
         [0, 0, 0, 1 / (Rint * Crm[5]), 1 / (Rint * Crm[5]), 0, 2 / (Rint * Crm[5]), -8 / (Rint * Crm[5]), 2 / (Rint * Crm[5]), 0,
          2 / (Rint * Crm[5])],  # rm6
         [0, 0, 0, 2 / (Rint * Crm[6]), 0, 0, 0, 2 / (Rint * Crm[6]), -8 / (Rint * Crm[6]), 2 / (Rint * Crm[6]), 2 / (Rint * Crm[6])],
         # rm7
         [0, 2 / (Re[2] * Crm[7]), 0, 2 / (Rint * Crm[7]), 0, 0, 0, 0, 2 / (Rint * Crm[7]),
          -2 / (Re[2] * Crm[7]) - 6 / (Rint * Crm[7]), 2 / (Rint * Crm[7])],  # rm8
         [0, 14 / (Re[2] * Crm[8]), 0, 0, 0, 2 / (Rint * Crm[8]), 2 / (Rint * Crm[8]), 2 / (Rint * Crm[8]), 2 / (Rint * Crm[8]),
          2 / (Rint * Crm[8]), -14 / (Re[2] * Crm[8]) - 10 / (Rint * Crm[8])]]  # rm9

    A = torch.tensor(A, dtype=torch.float32)

    assert torch.equal(rounded(A), rounded(building_n9.make_system_matrix()))


@pytest.mark.parametrize(
    'building',
    [
        pytest.lazy_fixture('building_n2'),
        pytest.lazy_fixture('building_n9'),
    ], )
def test_matrix_multiplication(building):
    bld = building

    B = bld.input_matrix()
    Tout = torch.tensor(15)
    Q = 5 * torch.ones(len(bld.rooms))
    u = bld.input_vector(Tout, Q)

    x = 20 * torch.ones((2 + len(bld.rooms), 1))
    A = bld.make_system_matrix()

    assert torch.Size([2 + len(bld.rooms), 1]) == (A @ x + B @ u).shape, "Should be a column vector"


@pytest.mark.parametrize(
    'building',
    [
        pytest.lazy_fixture('building_n2'),
        pytest.lazy_fixture('building_n9'),
    ], )
def test_input_matrix_shape(building):
    B = building.input_matrix()

    assert B.shape == torch.Size([2 + len(building.rooms), 1 + len(building.rooms)])


if __name__ == '__main__':
    pytest.main()

    print("__main__ reached")
