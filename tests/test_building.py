import numpy as np
import pytest
import torch

from rcmodel.physical import Building
from rcmodel.physical.room import Room
from rcmodel.tools.helper_functions import change_origin


def rounded(x):
    """torch doesent round to n decimal place so this function does that"""
    n_digits = 4
    return torch.round(x * 10**n_digits) / (10**n_digits)


# -------- Tests --------


def test_rounded():
    x = torch.tensor(
        [[0.71937193, 0.87787193, 0.40067193], [0.90667193, 0.12417193, 0.32707193], [0.71237193, 0.27967193, 0.01457193]]
    )
    y = torch.tensor([[0.7194, 0.8779, 0.4007], [0.9067, 0.1242, 0.3271], [0.7124, 0.2797, 0.0146]])
    assert torch.equal(rounded(x), y), "function not part of the model but is used for testing"


@pytest.mark.parametrize(
    "building, num_walls", [(pytest.lazy_fixture("building_n2"), 7), (pytest.lazy_fixture("building_n9"), 27)]
)
def test_unique_walls(building, num_walls):
    assert len(building.Walls) == num_walls, "Num walls counted incorrectly"


@pytest.mark.parametrize(
    "building, external_area", [(pytest.lazy_fixture("building_n2"), 30), (pytest.lazy_fixture("building_n9"), 44.180)]
)
def test_surf_area(building, external_area):
    assert round(building.surf_area.item(), 3) == external_area


@pytest.mark.parametrize(
    "building, K",
    [
        (pytest.lazy_fixture("building_n2"), torch.tensor([[0.0, 30.0, 30.0], [30.0, 0.0, 50.0], [30.0, 50.0, 0.0]])),
        (
            pytest.lazy_fixture("building_n9"),
            torch.tensor(
                [
                    [0.0000, 20.0000, 10.0000, 22.3607, 4.0000, 0.0000, 0.0000, 0.0000, 4.0000, 28.0000],
                    [20.0000, 0.0000, 50.0000, 50.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                    [10.0000, 50.0000, 0.0000, 50.0000, 0.0000, 0.0000, 10.0000, 20.0000, 20.0000, 0.0000],
                    [22.3607, 50.0000, 50.0000, 0.0000, 20.0000, 20.0000, 10.0000, 0.0000, 0.0000, 0.0000],
                    [4.0000, 0.0000, 0.0000, 20.0000, 0.0000, 20.0000, 0.0000, 0.0000, 0.0000, 20.0000],
                    [0.0000, 0.0000, 0.0000, 20.0000, 20.0000, 0.0000, 20.0000, 0.0000, 0.0000, 20.0000],
                    [0.0000, 0.0000, 10.0000, 10.0000, 0.0000, 20.0000, 0.0000, 20.0000, 0.0000, 20.0000],
                    [0.0000, 0.0000, 20.0000, 0.0000, 0.0000, 0.0000, 20.0000, 0.0000, 20.0000, 20.0000],
                    [4.0000, 0.0000, 20.0000, 0.0000, 0.0000, 0.0000, 0.0000, 20.0000, 0.0000, 20.0000],
                    [28.0000, 0.0000, 0.0000, 0.0000, 20.0000, 20.0000, 20.0000, 20.0000, 20.0000, 0.0000],
                ]
            ),
        ),
    ],
)
def test_connectivity_matrix(building, K):
    k = building.make_thermal_conductivity_matrix()

    assert torch.equal(rounded(K), rounded(k))


def test_make_system_matrix_rooms_n2(building_n2):
    Re = building_n2.Re
    Ce = building_n2.Ce
    Rint = building_n2.Rint
    Crm1 = building_n2.rooms[0].capacitance
    Crm2 = building_n2.rooms[1].capacitance

    A = [
        [-30 / (Re[0] * Ce[0]) - 30 / (Re[1] * Ce[0]), 30 / (Re[1] * Ce[0]), 0, 0],
        [30 / (Re[1] * Ce[1]), -30 / (Re[1] * Ce[1]) - 30 / (Re[2] * Ce[1]), 15 / (Re[2] * Ce[1]), 15 / (Re[2] * Ce[1])],
        [0, 15 / (Re[2] * Crm1), -15 / (Re[2] * Crm1) - 5 / (Rint * Crm1), 5 / (Rint * Crm1)],
        [0, 15 / (Re[2] * Crm2), 5 / (Rint * Crm2), -15 / (Re[2] * Crm2) - 5 / (Rint * Crm2)],
    ]

    A = torch.tensor(A, dtype=torch.float32)

    assert torch.equal(rounded(A), rounded(building_n2.make_system_matrix()))


def test_make_system_matrix_rooms_n9(building_n9):
    Re = building_n9.Re
    Ce = building_n9.Ce
    Rint = building_n9.Rint

    Crm = [room.capacitance for room in building_n9.rooms]

    sa = building_n9.surf_area.item()  # Need .item() otherwise error

    A = [
        [-sa / (Re[0] * Ce[0]) - sa / (Re[1] * Ce[0]), sa / (Re[1] * Ce[0]), 0, 0, 0, 0, 0, 0, 0, 0, 0],  # ex1
        [
            sa / (Re[1] * Ce[1]),
            -sa / (Re[1] * Ce[1]) - sa / (Re[2] * Ce[1]),
            10 / (Re[2] * Ce[1]),
            5 / (Re[2] * Ce[1]),
            np.sqrt(10**2 + 5**2) / (Re[2] * Ce[1]),
            2 / (Re[2] * Ce[1]),
            0,
            0,
            0,
            2 / (Re[2] * Ce[1]),
            14 / (Re[2] * Ce[1]),
        ],  # ex2
        [
            0,
            10 / (Re[2] * Crm[0]),
            -10 / (Re[2] * Crm[0]) - 10 / (Rint * Crm[0]),
            5 / (Rint * Crm[0]),
            5 / (Rint * Crm[0]),
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # rm1
        [
            0,
            5 / (Re[2] * Crm[1]),
            5 / (Rint * Crm[1]),
            -5 / (Re[2] * Crm[1]) - 15 / (Rint * Crm[1]),
            5 / (Rint * Crm[1]),
            0,
            0,
            1 / (Rint * Crm[1]),
            2 / (Rint * Crm[1]),
            2 / (Rint * Crm[1]),
            0,
        ],  # rm2
        [
            0,
            np.sqrt(10**2 + 5**2) / (Re[2] * Crm[2]),
            5 / (Rint * Crm[2]),
            5 / (Rint * Crm[2]),
            -np.sqrt(10**2 + 5**2) / (Re[2] * Crm[2]) - 15 / (Rint * Crm[2]),
            2 / (Rint * Crm[2]),
            2 / (Rint * Crm[2]),
            1 / (Rint * Crm[2]),
            0,
            0,
            0,
        ],  # rm3
        [
            0,
            2 / (Re[2] * Crm[3]),
            0,
            0,
            2 / (Rint * Crm[3]),
            -2 / (Re[2] * Crm[3]) - 6 / (Rint * Crm[3]),
            2 / (Rint * Crm[3]),
            0,
            0,
            0,
            2 / (Rint * Crm[3]),
        ],  # rm4
        [
            0,
            0,
            0,
            0,
            2 / (Rint * Crm[4]),
            2 / (Rint * Crm[4]),
            -8 / (Rint * Crm[4]),
            2 / (Rint * Crm[4]),
            0,
            0,
            2 / (Rint * Crm[4]),
        ],
        # rm5
        [
            0,
            0,
            0,
            1 / (Rint * Crm[5]),
            1 / (Rint * Crm[5]),
            0,
            2 / (Rint * Crm[5]),
            -8 / (Rint * Crm[5]),
            2 / (Rint * Crm[5]),
            0,
            2 / (Rint * Crm[5]),
        ],  # rm6
        [
            0,
            0,
            0,
            2 / (Rint * Crm[6]),
            0,
            0,
            0,
            2 / (Rint * Crm[6]),
            -8 / (Rint * Crm[6]),
            2 / (Rint * Crm[6]),
            2 / (Rint * Crm[6]),
        ],
        # rm7
        [
            0,
            2 / (Re[2] * Crm[7]),
            0,
            2 / (Rint * Crm[7]),
            0,
            0,
            0,
            0,
            2 / (Rint * Crm[7]),
            -2 / (Re[2] * Crm[7]) - 6 / (Rint * Crm[7]),
            2 / (Rint * Crm[7]),
        ],  # rm8
        [
            0,
            14 / (Re[2] * Crm[8]),
            0,
            0,
            0,
            2 / (Rint * Crm[8]),
            2 / (Rint * Crm[8]),
            2 / (Rint * Crm[8]),
            2 / (Rint * Crm[8]),
            2 / (Rint * Crm[8]),
            -14 / (Re[2] * Crm[8]) - 10 / (Rint * Crm[8]),
        ],
    ]  # rm9

    A = torch.tensor(A, dtype=torch.float32)

    assert torch.equal(rounded(A), rounded(building_n9.make_system_matrix()))


@pytest.mark.parametrize(
    "building",
    [
        pytest.lazy_fixture("building_n2"),
        pytest.lazy_fixture("building_n9"),
    ],
)
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
    "building",
    [
        pytest.lazy_fixture("building_n2"),
        pytest.lazy_fixture("building_n9"),
    ],
)
def test_input_matrix_shape(building):
    B = building.input_matrix()

    assert B.shape == torch.Size([2 + len(building.rooms), 1 + len(building.rooms)])


# -------- Geometry: units and areas --------
# Building.__init__ rewrites room.walls into indices into its own wall list, so a Room cannot
# be reused across two Buildings. Each test below builds its own.


def test_room_area_is_floor_area_not_perimeter():
    """Room.area must be the enclosed floor area (m^2), which is what scales C_rm and converts
    cool/gain from W/m2 to W.

    scipy names these the other way round for a 2D ConvexHull: .volume is the enclosed area and
    .area is the perimeter. Reading .area made a 3 x 4 m room report 14.0 instead of 12.0.
    """
    rm = Room("3x4", [[0, 0], [0, 4], [3, 4], [3, 0]])

    assert rm.area == pytest.approx(12.0), "area should be 3*4=12 m^2, not the perimeter 14 m"


def test_room_area_of_concave_room_uses_convex_hull():
    """Known simplification, pinned rather than fixed: ConvexHull fills in a concave room, while
    Room.walls keeps the original vertex order. So an L-shaped room's walls and its area come
    from different polygons - the area is the hull's 14 m^2, not the L's true 12 m^2.
    """
    l_shape = [[0, 0], [0, 4], [2, 4], [2, 2], [4, 2], [4, 0]]
    rm = Room("L", l_shape)

    assert rm.area == pytest.approx(14.0), "convex hull fills the notch: 4*4 - 2*1 = 14"
    assert len(rm.walls) == len(l_shape), "walls still follow the original concave outline"


def test_change_origin_is_in_metres():
    """change_origin only shifts to a common origin - no unit conversion. It used to divide by
    10, so configs had to be written in decimetres to come out as metres.
    """
    rooms = change_origin([[[10.0, 20.0], [10.0, 24.0], [13.0, 24.0], [13.0, 20.0]]])

    assert rooms == [[[0.0, 0.0], [0.0, 4.0], [3.0, 4.0], [3.0, 0.0]]]


def test_change_origin_preserves_relative_position():
    """Two rooms keep their offset from each other; only the shared origin moves."""
    rooms = change_origin([[[5.0, 5.0], [5.0, 7.0], [7.0, 7.0], [7.0, 5.0]], [[7.0, 5.0], [9.0, 5.0]]])

    assert rooms[0][0] == [0.0, 0.0], "bounding-box corner becomes the origin"
    assert rooms[1] == [[2.0, 0.0], [4.0, 0.0]], "second room keeps its offset from the first"


def test_surf_area_uses_room_height():
    """surf_area is external wall area: perimeter * height. The EnergyPlus 1-zone case is a
    15.24 m square, 4.572 m tall -> 4 * 15.24 * 4.572 = 278.71 m^2.
    """
    coordinates = [[0, 0], [0, 15.24], [15.24, 15.24], [15.24, 0]]

    bld = Building([Room("zone_one", coordinates)], 4.572)
    assert bld.surf_area.item() == pytest.approx(278.71, abs=0.01)

    # height defaults to 1, which makes surf_area the bare perimeter.
    bare = Building([Room("zone_one", coordinates)])
    assert bare.surf_area.item() == pytest.approx(60.96, abs=0.01)


if __name__ == "__main__":
    pytest.main()

    print("__main__ reached")
