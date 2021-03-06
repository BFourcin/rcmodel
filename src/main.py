import ray
import pandas as pd
import numpy as np
import torch
from rcmodel import *

if __name__ == '__main__':

    use_ray = False

    epochs = 2

    if use_ray:
        num_cpus = 4
        num_jobs = num_cpus
        ray.init(num_cpus=num_cpus)

    # Initialise scaling class
    rm_CA = [100, 1e4]  # [min, max] Capacitance/area
    ex_C = [1e3, 1e8]  # Capacitance
    R = [0.1, 5]  # Resistance ((K.m^2)/W)
    Q_limit = [-300, 300]  # Cooling limit and gain limit in W/m2
    scaling = InputScaling(rm_CA, ex_C, R, Q_limit)

    # Laptop:
    weather_data_path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/Met Office Weather ' \
                        'Files/JuneSept.csv'
    csv_path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/DummyData/test2d_sorted.csv'

    # Hydra:
    # weather_data_path = '/home/benf/LSI/Data/Met Office Weather Files/JuneSept.csv'
    # csv_path = '/home/benf/LSI/Data/DummyData/train5d_sorted.csv'

    # policy = PriorCoolingPolicy()
    policy = PolicyNetwork(5, 2)
    model = initialise_model(policy, scaling, weather_data_path)

    model.loads = torch.nn.Parameter(model.loads/model.loads * torch.logit(torch.tensor([[0.1], [0.003]])))

    if use_ray:
        RayActor = ray.remote(RayActor)
        actors = [RayActor.remote(model, csv_path, physical_training=True, policy_training=False) for _ in range(num_jobs)]

        results = ray.get([a.worker.remote(num, epochs) for num, a in enumerate(actors)])
        ray.shutdown()

    else:
        actor = RayActor(model, csv_path)
        actor.physical_training = False  # Turn physical optimisation off
        actor.policy_training = True  # Turn cooling optimisation off
        results = actor.worker(0, epochs)

    params_heading = ['Rm Cap/m2 (J/K.m2)', 'Ext Wl Cap 1 (J/K)', 'Ext Wl Cap 2 (J/K)', 'Ext Wl Res 1 (K.m2/W)',
                      'Ext Wl Res 2 (K.m2/W)', 'Ext Wl Res 3 (K.m2/W)', 'Int Wl Res (K.m2/W)', 'Offset Gain (W/m2)']
    cooling_heading = ['Cooling (W)']
    headings = [['Run Number'], params_heading, cooling_heading,
                ['Final Average Train Loss', 'Final Avg Test Loss']]
    flat_list = [item for sublist in headings for item in sublist]

    if use_ray:
        df = pd.DataFrame(np.array(results), columns=flat_list)
    else:
        df = pd.DataFrame([np.array(results)], columns=flat_list)

    df.to_csv('./outputs/results.csv', index=False, )




    # policy = PriorCoolingPolicy(temp_on=23, temp_off=21, time_on=9, time_off=19)
    # weather_data_path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/Met Office Weather Files/JuneSept.csv'
    #
    # def init_scaling():
    #     # Initialise scaling class
    #     rm_CA = [100, 1e4]  # [min, max] Capacitance/area
    #     ex_C = [1e3, 1e8]  # Capacitance
    #     R = [0.1, 5]  # Resistance ((K.m^2)/W)
    #     Q_limit = [300]  # Cooling limit and gain limit in W/m2
    #     scaling = InputScaling(rm_CA, ex_C, R, Q_limit)
    #     return scaling
    #
    # # Initialise Model
    # scaling = init_scaling()
    # # policy = PolicyNetwork(5, 2)
    # model = initialise_model(policy, scaling, weather_data_path)
    #
    # # model.load('./rcmodel1000.pt')
    #
    # policy = model.cooling_policy
    #
    # # import torch
    # # with torch.no_grad():
    # #     sample_size = 24 * 60 ** 2 / 30  # ONE DAY
    # #     csv_path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/DummyData/train5d_sorted.csv'
    # #     plot_results_data = BuildingTemperatureDataset(csv_path, 5 * sample_size, all=True)
    # #     plot_dataloader = torch.utils.data.DataLoader(plot_results_data, batch_size=1, shuffle=False)
    # #     pltsolution_1rm(model, plot_dataloader)
    #
    # import time
    # s = time.time()
    # policy_image(policy)
    # print(time.time()-s)
    # # import cProfile
    # #
    # # cProfile.run('policy_image(policy)', sort='tottime')


