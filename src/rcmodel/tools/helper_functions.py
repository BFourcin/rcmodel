from ..room import Room
from ..building import Building
from ..RCModel import RCModel
from ..optimisation import OptimiseRC
from xitorch.interpolate import Interp1D

import pandas as pd
import torch


def initialise_model(pi, scaling, weather_data_path):

    def change_origin(coords):
        x0 = 92.07
        y0 = 125.94

        for i in range(len(coords)):
            coords[i][0] = round((coords[i][0] - x0) / 10, 2)
            coords[i][1] = round((coords[i][1] - y0) / 10, 2)

        return coords

    rooms = []

    name = "seminar_rm_a_t0106"
    coords = change_origin(
        [[92.07, 125.94], [92.07, 231.74], [129.00, 231.74], [154.45, 231.74], [172.64, 231.74], [172.64, 125.94]])
    rooms.append(Room(name, coords))

    # Initialise Building
    bld = Building(rooms)

    df = pd.read_csv(weather_data_path)
    Tout = torch.tensor(df['Hourly Temperature (°C)'])
    t = torch.tensor(df['time'])

    Tout_continuous = Interp1D(t, Tout, method='linear')

    # Initialise RCModel with the building
    transform = torch.sigmoid
    model = RCModel(bld, scaling, Tout_continuous, transform, pi)

    return model


def initialise_prior(scaling, weather_data_path):

    # policy = PolicyNetwork(7,2)
    prior = PriorCoolingPolicy()

    model = initialise_model(prior, scaling, weather_data_path)

    dt = 30
    sample_size = 24 * 60 ** 2 / dt

    op = OptimiseRC(model, csv_path, sample_size, dt, lr=1e-3, opt_id=opt_id)