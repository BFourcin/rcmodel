import pytest
import torch
import numpy as np

from rcmodel.room import Room
from rcmodel.RCModel import RCModel


#simple func to initilaise a class
def getbuilding(listrms):
    height = 1
    Re = [5, 1, 0.5]
    Ce = [1e3,8e2]
    Rint = 0.1

    bld = RCModel(listrms, height, Re, Ce, Rint)

    return bld


#fixture for using multiple room tests
@pytest.fixture(params=[pytest.lazy_fixture('rooms')])
def building(request):
    height = 1
    Re = [5, 1, 0.5]
    Ce = [1e3,8e2]
    Rint = 0.1


    bld = RCModel(request.param, height, Re, Ce, Rint)

    return bld


def rounded(x):
    """torch doesent round to n decimal place so this function does that"""
    n_digits = 4
    return torch.round(x * 10**n_digits) / (10**n_digits)



###-------- Tests --------###

def test_rounded():
    x = torch.tensor([[0.71937193, 0.87787193, 0.40067193],
                    [0.90667193, 0.12417193, 0.32707193],
                    [0.71237193, 0.27967193, 0.01457193]])
    y = torch.tensor([[0.7194, 0.8779, 0.4007],
                    [0.9067, 0.1242, 0.3271],
                    [0.7124, 0.2797, 0.0146]])
    assert torch.equal(rounded(x),y), "function not part of the model but is used for testing"


@pytest.mark.parametrize(
    'rooms, numwalls',
    [
    (pytest.lazy_fixture('rooms_n2'), 7),
    (pytest.lazy_fixture('rooms_n9'), 27)
    ])
def test_unique_walls(building, rooms, numwalls):

    assert len(building.Walls) == numwalls, "Num walls counted incorrectly"


@pytest.mark.parametrize(
    'rooms, external_area',
    [
    (pytest.lazy_fixture('rooms_n2'), 30),
    (pytest.lazy_fixture('rooms_n9'), 44.180)
    ])
def test_surf_area(building, rooms, external_area):

    assert round(building.surf_area.item(), 3) == external_area


@pytest.mark.parametrize(
    'rooms, K',
    [
    (pytest.lazy_fixture('rooms_n2'), torch.tensor([[ 0., 30., 30.],
                                                    [30.,  0., 50.],
                                                    [30., 50.,  0.]])),
    (pytest.lazy_fixture('rooms_n9'), torch.tensor([[ 0.0000, 20.0000, 10.0000, 22.3607,  4.0000,  0.0000,  0.0000,  0.0000, 4.0000, 28.0000],
                                                    [20.0000,  0.0000, 50.0000, 50.0000,  0.0000,  0.0000,  0.0000,  0.0000, 0.0000,  0.0000],
                                                    [10.0000, 50.0000,  0.0000, 50.0000,  0.0000,  0.0000, 10.0000, 20.0000, 20.0000,  0.0000],
                                                    [22.3607, 50.0000, 50.0000,  0.0000, 20.0000, 20.0000, 10.0000,  0.0000, 0.0000,  0.0000],
                                                    [ 4.0000,  0.0000,  0.0000, 20.0000,  0.0000, 20.0000,  0.0000,  0.0000, 0.0000, 20.0000],
                                                    [ 0.0000,  0.0000,  0.0000, 20.0000, 20.0000,  0.0000, 20.0000,  0.0000, 0.0000, 20.0000],
                                                    [ 0.0000,  0.0000, 10.0000, 10.0000,  0.0000, 20.0000,  0.0000, 20.0000, 0.0000, 20.0000],
                                                    [ 0.0000,  0.0000, 20.0000,  0.0000,  0.0000,  0.0000, 20.0000,  0.0000, 20.0000, 20.0000],
                                                    [ 4.0000,  0.0000, 20.0000,  0.0000,  0.0000,  0.0000,  0.0000, 20.0000, 0.0000, 20.0000],
                                                    [28.0000,  0.0000,  0.0000,  0.0000, 20.0000, 20.0000, 20.0000, 20.0000, 20.0000,  0.0000]])),
    ])
def test_connectivity_matrix(building, rooms, K):

    k = building.make_connectivity_matrix()

    assert torch.equal(rounded(K),rounded(k))










def test_make_system_matrix_rooms_n2(rooms_n2):
    height = 1
    Re = [5, 1, 0.5]
    Ce = [1e3,8e2]
    Rint = 0.1

    b = RCModel(rooms_n2, height, Re, Ce, Rint)

    A = [[-30/(Re[0]*Ce[0]) -30/(Re[1]*Ce[0]), 30/(Re[1]*Ce[0]), 0, 0],
         [30/(Re[1]*Ce[1]), -30/(Re[1]*Ce[1]) -30/(Re[2]*Ce[1]), 15/(Re[2]*Ce[1]), 15/(Re[2]*Ce[1])],
         [0, 15/(Re[2]*1e3), -15/(Re[2]*1e3)- 5/(Rint*1e3), 5/(Rint*1e3)],
         [0, 15/(Re[2]*2e3), 5/(Rint*2e3), -15/(Re[2]*2e3) -5/(Rint*2e3)]]

    A = torch.tensor(A, dtype=torch.float32)

    assert torch.equal(rounded(A), rounded(b.make_system_matrix()))


def test_make_system_matrix_rooms_n9(rooms_n9):
    height = 1
    Re = [5, 1, 0.5]
    Ce = [1e3,8e2]
    Rint = 0.1

    b = RCModel(rooms_n9, height, Re, Ce, Rint)

    sa = b.surf_area.item() #Need .item() otherwise error

    A = [[-sa/(Re[0]*Ce[0]) -sa/(Re[1]*Ce[0]), sa/(Re[1]*Ce[0]), 0, 0, 0, 0, 0, 0, 0, 0, 0],#ex1
         [sa/(Re[1]*Ce[1]), -sa/(Re[1]*Ce[1]) -sa/(Re[2]*Ce[1]), 10/(Re[2]*Ce[1]), 5/(Re[2]*Ce[1]), np.sqrt(10**2+5**2)/(Re[2]*Ce[1]), 2/(Re[2]*Ce[1]), 0, 0, 0, 2/(Re[2]*Ce[1]), 14/(Re[2]*Ce[1])],#ex2
         [0, 10/(Re[2]*1e3), -10/(Re[2]*1e3)- 10/(Rint*1e3), 5/(Rint*1e3), 5/(Rint*1e3), 0, 0, 0, 0, 0, 0],#rm1
         [0, 5/(Re[2]*2e3), 5/(Rint*2e3), -5/(Re[2]*2e3) -15/(Rint*2e3), 5/(Rint*2e3), 0, 0, 1/(Rint*2e3), 2/(Rint*2e3), 2/(Rint*2e3), 0],#rm2
         [0, np.sqrt(10**2+5**2)/(Re[2]*3e3), 5/(Rint*3e3), 5/(Rint*3e3), -np.sqrt(10**2+5**2)/(Re[2]*3e3) -15/(Rint*3e3), 2/(Rint*3e3), 2/(Rint*3e3), 1/(Rint*3e3), 0, 0, 0],#rm3
         [0, 2/(Re[2]*4e3), 0, 0, 2/(Rint*4e3), -2/(Re[2]*4e3)-6/(Rint*4e3), 2/(Rint*4e3), 0, 0, 0, 2/(Rint*4e3)],#rm4
         [0, 0, 0, 0, 2/(Rint*5e3), 2/(Rint*5e3), -8/(Rint*5e3), 2/(Rint*5e3), 0, 0, 2/(Rint*5e3)],#rm5
         [0, 0, 0, 1/(Rint*6e3), 1/(Rint*6e3), 0, 2/(Rint*6e3), -8/(Rint*6e3), 2/(Rint*6e3), 0, 2/(Rint*6e3)],#rm6
         [0, 0, 0, 2/(Rint*7e3), 0, 0, 0, 2/(Rint*7e3), -8/(Rint*7e3), 2/(Rint*7e3), 2/(Rint*7e3)],#rm7
         [0, 2/(Re[2]*8e3), 0, 2/(Rint*8e3), 0, 0, 0, 0, 2/(Rint*8e3), -2/(Re[2]*8e3) -6/(Rint*8e3), 2/(Rint*8e3)],#rm8
         [0, 14/(Re[2]*9e3), 0, 0, 0, 2/(Rint*9e3), 2/(Rint*9e3), 2/(Rint*9e3), 2/(Rint*9e3), 2/(Rint*9e3), -14/(Re[2]*9e3)-10/(Rint*9e3)]]#rm9

    A = torch.tensor(A, dtype=torch.float32)

    assert torch.equal(rounded(A), rounded(b.make_system_matrix()))



def test_matrix_multiplication(rooms_n9):

    bld = getbuilding(rooms_n9)

    B = bld.input_matrix()
    Tout = 15
    Q = 5 * torch.ones(len(bld.rooms))
    u = bld.input_vector(Tout, Q)

    x = 20*torch.ones((2+len(bld.rooms),1))
    A = bld.make_system_matrix()

    assert torch.Size([2 + len(bld.rooms), 1]) == (A@x+B@u).shape, "Should be a column vector"


@pytest.mark.parametrize(
    'rooms',
    [
    (pytest.lazy_fixture('rooms_n2')),
    (pytest.lazy_fixture('rooms_n9'))
    ])
def test_input_matrix_shape(building, rooms):

    B = building.input_matrix()

    assert B.shape == torch.Size([2+len(building.rooms), 1+len(building.rooms)])
