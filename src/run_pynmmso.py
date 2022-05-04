import torch
from xitorch.interpolate import Interp1D
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import time
from pynmmso import Nmmso
from pynmmso import MultiprocessorFitnessCaller
from pynmmso.listeners import TraceListener

from rcmodel import *


def make_original_model():
    # ----- Create Original model and produce a dummy data csv file: -----

    # Initialise scaling class
    rm_CA = [100, 1e4]  # [min, max] Capacitance/area
    ex_C = [1e3, 1e8]  # Capacitance
    R = [0.1, 5]  # Resistance ((K.m^2)/W)
    Q_limit = [20]  # Cooling limit and gain limit in W/m2
    scaling = InputScaling(rm_CA, ex_C, R, Q_limit)

    # Laptop
    weather_data_path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/Met Office Weather Files/JuneSept.csv'

    # Hydra:
    # weather_data_path = '/home/benf/LSI/Data/Met Office Weather Files/JuneSept.csv'

    policy = None
    # prior = PriorCoolingPolicy()

    model_origin = initialise_model(policy, scaling, weather_data_path)
    # model_origin.state_dict()['params'][7] = 0.001  # manually lower 'gain'

    # load a model which we will then try to replicate.
    load_policy = None
    load_physical = './return_model.pt'

    if load_policy:
        model_origin.load(load_policy)  # load policy
        model_origin.init_physical()  # re-randomise physical params, as they were also copied from the loaded policy

    if load_physical:
        # m = initialise_model(None, scaling, weather_data_path)  # dummy model to load physical params on to.
        m = initialise_model(None, scaling, weather_data_path)  # dummy model to load physical params on to.
        m.load(load_physical)
        model_origin.params = m.params  # put loaded physical parameters onto model.
        model_origin.loads = m.loads
        del m

    with open('original_params.txt', 'w') as f:
        f.write(str(model_origin.scaling.physical_param_scaling(model_origin.transform(model_origin.params))))

    t_eval = torch.arange(0, 1 * 24 * 60 ** 2, 30, dtype=torch.float64) + 1622505600  # + min start time of weather data

    output_path = './return_to_sender_out_sorted.csv'

    pred = model_to_csv(model_origin, t_eval, output_path)

    iv_array = Interp1D(t_eval, pred.T.detach(), method='linear')

    # see what the original model looks like:
    pltsolution_1rm(model_origin, prediction=pred, time=t_eval, filename='./original.png')

    return iv_array


class MyProblem:
    def __init__(self, model, dataset, time_data):
        self.model = model
        self.dataset = dataset
        self.time_data = time_data

        self.op = OptimiseRC(self.model, self.dataset, self.dataset, lr=1e-2, opt_id=0)

    def fitness(self, params):
        n = self.model.building.n_params
        self.model.params = torch.nn.Parameter(torch.tensor(params[0:n], dtype=torch.float32))
        loads = torch.tensor(np.array([params[n:n + 2 * len(self.model.building.rooms)]]), dtype=torch.float32)
        self.model.loads = torch.nn.Parameter(loads.reshape(2, len(self.model.building.rooms)))

        # self.model.iv_array = self.model.get_iv_array(self.time_data)  # find the latent variables
        avg_train_loss = self.op.train()

        return -avg_train_loss  # negative because pynmmso maximises

    def get_bounds(self):
        n = self.model.building.n_params + 2 * len(self.model.building.rooms)
        return list(np.zeros(n)), list(np.ones(n))


def main(dt_string):
    # Get data:
    # Laptop
    weather_data_path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/Met Office Weather Files/JuneSept.csv'

    # Hydra:
    # weather_data_path = '/home/benf/LSI/Data/Met Office Weather Files/JuneSept.csv'

    # csv_path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/DummyData/train5d_sorted.csv'
    csv_path = './return_to_sender_out_sorted.csv'
    iv_array = make_original_model()

    dt = 30  # timestep (seconds), data and the model are sampled at this frequency
    sample_size = int(24 * 60 ** 2 / dt)
    train_dataset = BuildingTemperatureDataset(csv_path, sample_size, all=True)
    time_data = torch.tensor(pd.read_csv(csv_path, skiprows=0).iloc[:, 1], dtype=torch.float64)
    temp_data = torch.tensor(pd.read_csv(csv_path, skiprows=0).iloc[:, 2:].to_numpy(dtype=np.float32),
                             dtype=torch.float32)

    def init_scaling():
        # Initialise scaling class
        rm_CA = [100, 1e4]  # [min, max] Capacitance/area
        ex_C = [1e3, 1e8]  # Capacitance
        R = [0.1, 5]  # Resistance ((K.m^2)/W)
        Q_limit = [20]  # Cooling limit and gain limit in W/m2
        scaling = InputScaling(rm_CA, ex_C, R, Q_limit)
        return scaling

    # Initialise Model
    scaling = init_scaling()
    policy = PolicyNetwork(5, 2)
    model = initialise_model(policy, scaling, weather_data_path)
    model.Tin_continuous = Interp1D(time_data, temp_data[:, 0:len(model.building.rooms)].T, method='linear')

    model.iv_array = iv_array

    # a = MyProblem(model, train_dataset, time_data)
    # a.fitness(a.get_bounds()[0])

    # ----------nmmso----------
    number_of_fitness_evaluations = 30000
    num_workers = 8

    my_multi_processor_fitness_caller = MultiprocessorFitnessCaller(num_workers)

    # with MultiprocessorFitnessCaller(num_workers) as my_multi_processor_fitness_caller:
    nmmso = Nmmso(MyProblem(model, train_dataset, time_data), fitness_caller=my_multi_processor_fitness_caller)
    nmmso.add_listener(TraceListener(level=2))
    my_result = nmmso.run(number_of_fitness_evaluations)

    my_multi_processor_fitness_caller.finish()

    # nmmso = Nmmso(MyProblem(model, train_dataset, time_data))
    # nmmso.add_listener(TraceListener(level=5))
    # my_result = nmmso.run(number_of_fitness_evaluations)

    for mode_result in my_result:
        print("Mode at {} has value {}".format(mode_result.location, mode_result.value))

    # ------save results to csv------
    loc = []
    val = []
    for mode_result in my_result:
        loc.append(mode_result.location)
        val.append(mode_result.value)

    loc = np.array(loc)
    val = np.array(val)
    output = np.concatenate((loc, np.vstack(val)), axis=1)

    df = pd.DataFrame(output)
    df.to_csv('logs/' + dt_string + '_results_nmmso' + '.csv', index=False)


if __name__ == "__main__":

    from datetime import datetime
    from pathlib import Path
    import sys
    import os

    # ---Add Logging file---

    # datetime object containing current date and time
    now = datetime.now()
    # YY/mm/dd H:M:S
    dt_string = now.strftime("%y-%m-%d_%H:%M:%S")

    logfile = 'logs/' + dt_string + '_log_nmmso' + '.log'

    # Create dir if needed
    Path("./logs/").mkdir(parents=True, exist_ok=True)

    # Add Logging file:
    if os.path.exists(logfile):
        os.remove(logfile)

    # set buffering to none, means that result is written to log in realtime. Otherwise this is done at the end.
    log = open(logfile, "a", buffering=1)
    sys.stdout = log  # print() now goes to .log

    # ---Start Run---
    start = time.time()
    main(dt_string)
    print("duration =", time.time() - start)

