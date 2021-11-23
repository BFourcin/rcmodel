import torch
import pandas as pd
from .room import Room
from .building import Building
from .RCModel import RCModel
from .tools import InputScaling
from xitorch.interpolate import Interp1D


def initialise_model(pi, scaling, weather_data_path):
    torch.cuda.is_available = lambda: False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def change_origin(coords):
        x0 = 92.07
        y0 = 125.94

        for i in range(len(coords)):
            coords[i][0] = round((coords[i][0] - x0) / 10, 2)
            coords[i][1] = round((coords[i][1] - y0) / 10, 2)

        return coords
    capacitance = 3000  # Variable changed later
    rooms = []

    name = "seminar_rm_a_t0106"
    coords = change_origin(
        [[92.07, 125.94], [92.07, 231.74], [129.00, 231.74], [154.45, 231.74], [172.64, 231.74], [172.64, 125.94]])
    rooms.append(Room(name, capacitance, coords))

    # Initialise Building
    height = 1
    Re = [4, 1, 0.55]  # Sum of R makes Uval=0.18 #Variable changed later
    Ce = [1.2 * 10 ** 3, 0.8 * 10 ** 3]  # Variable changed later
    Rint = 0.66  # Uval = 1/R = 1.5 #Variable changed later

    bld = Building(rooms, height, Re, Ce, Rint)

    df = pd.read_csv(weather_data_path)
    Tout = torch.tensor(df['Hourly Temperature (°C)'], device=device)
    t = torch.tensor(df['time'], device=device)

    Tout_continuous = Interp1D(t, Tout, method='linear')

    # Initialise RCModel with the building
    transform = torch.sigmoid
    model = RCModel(bld, scaling, Tout_continuous, transform, pi)
    model.to(device)  # put model on GPU if available
    model.Q_lim = 10000

    return model


def example_optimise():

    model = initialise_model(None)
    csv_path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/DummyData/train5d_sorted.csv'
    dt = 30
    sample_size = 2*60**2/dt
    start = time.time()
    op = OptimiseRC(model, csv_path, sample_size, dt)

    epochs = 5

    for i in range(epochs):
        print(f"Epoch {i + 1}\n-------------------------------")
        op.train()

    print(f'duration {time.time()-start:.1f} s')


def example_optimise_ray():

    import ray

    model = initialise_model(None)
    csv_path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/DummyData/train5d_sorted.csv'
    dt = 30
    sample_size = 2 * 60 ** 2 / dt

    num_cpus = 5
    num_jobs = num_cpus

    ray.init(num_cpus=num_cpus)

    start = time.time()

    RemoteOptimise = ray.remote(OptimiseRC)  # need if don't use @ray.remote

    actors = [RemoteOptimise.remote(model, csv_path, sample_size, dt) for _ in range(num_jobs)]

    epochs = 2
    results = ray.get([a.train_loop.remote(epochs) for a in actors])

    time.sleep(3)  # Print is cut short without sleep

    ray.shutdown()


if __name__ == '__main__':

    import time
    from optimisation import OptimiseRC

    # example_optimise_ray()
    example_optimise()