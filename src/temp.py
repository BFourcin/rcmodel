import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm, trange
from xitorch.interpolate import Interp1D


from latent_sim import LatentSim

from rcmodel import *

# Laptop
weather_data_path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/Met Office Weather Files/JuneSept.csv'
csv_path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/DummyData/train5d_sorted.csv'

# Hydra:
# weather_data_path = '/home/benf/LSI/Data/Met Office Weather Files/JuneSept.csv'
# csv_path = '/home/benf/LSI/Data/DummyData/train5d_sorted.csv'  # where building data

# Load model?
load_model_path_physical = './rcmodel1010.pt'  # or None
load_model_path_policy = './rcmodel1010.pt'

# load_model_path_physical = './physical_model.pt'  # or None
# load_model_path_policy = './prior_policy.pt'

# opt_id = 0

def worker(opt_id):
    torch.set_num_threads(1)

    dir_path = f'./outputs/run{opt_id}/models/'  # where to save

    def do_plots():
        # find smoothed curves:
        phys_smooth = exponential_smoothing(avg_train_loss_plot, conv_alpha, n=conv_win_len)
        pol_smooth = exponential_smoothing(rewards_plot, conv_alpha, n=conv_win_len)

        # Plot Loss:
        fig, axs = plt.subplots(figsize=(10, 7))
        ax2 = axs.twinx()
        ln1 = axs.plot(range(1, len(avg_train_loss_plot) + 1), avg_train_loss_plot, label='train loss')
        ln2 = axs.plot(range(1, len(avg_test_loss_plot) + 1), avg_test_loss_plot, label='test loss')
        ln3 = axs.plot(range(1, len(phys_smooth) + 1), phys_smooth, 'g', label='train loss - smoothed')
        c = []
        for i in range(len(phys_smooth)):
            c.append(convergence_criteria(phys_smooth[0:i + 1], conv_win_len))

        ln4 = ax2.plot(range(1, len(phys_smooth) + 1), c, 'k--', label='convergence')

        fig.suptitle('Train and Test Loss During Training', fontsize=16)
        axs.set_xlabel('Epoch')
        axs.set_ylabel('Mean Squared Error')
        axs.set_yscale('log')
        ax2.set_ylabel('Convergence')
        ax2.set_yscale('log')
        # legend:
        lns = ln1 + ln2 + ln3 + ln4
        labs = [l.get_label() for l in lns]
        axs.legend(lns, labs, loc=0)

        fig.savefig(f'./outputs/run{opt_id}/plots/LossPlot.png')
        plt.close()

        # Plot Rewards:
        fig, axs = plt.subplots(figsize=(10, 7))
        ax2 = axs.twinx()

        y = np.array(rewards_plot, dtype=np.float32)
        ln1 = axs.plot(range(1, len(y) + 1), -y, label='total rewards')
        ln2 = axs.plot(range(1, len(pol_smooth) + 1), -np.array(pol_smooth, dtype=np.float32), 'g',
                       label='rewards - smoothed')
        c = []
        for i in range(len(pol_smooth)):
            c.append(convergence_criteria(pol_smooth[0:i + 1], conv_win_len))

        ln3 = ax2.plot(range(1, len(pol_smooth) + 1), c, 'k--', label='convergence')

        # legend:
        lns = ln1 + ln2 + ln3
        labs = [l.get_label() for l in lns]
        axs.legend(lns, labs, loc=0)

        fig.suptitle('Rewards', fontsize=16)
        axs.set_xlabel('Epoch')
        axs.set_ylabel('Reward')
        axs.set_yscale('log')

        ax2.set_ylabel('Convergence')
        ax2.set_yscale('log')

        fig.savefig(f'./outputs/run{opt_id}/plots/RewardsPlot.png')
        plt.close()

    def init_scaling():
        # Initialise scaling class
        rm_CA = [100, 1e4]  # [min, max] Capacitance/area
        ex_C = [1e3, 1e8]  # Capacitance
        R = [0.1, 5]  # Resistance ((K.m^2)/W)
        Q_limit = [-300, 300]  # Cooling limit and gain limit in W/m2
        scaling = InputScaling(rm_CA, ex_C, R, Q_limit)
        return scaling

    # Initialise Model
    scaling = init_scaling()
    policy = PolicyNetwork(5, 2)
    model = initialise_model(policy, scaling, weather_data_path)

    # # Initialise Optimise Class - for training physical model
    dt = 30
    sample_size = 24 * 60 ** 2 / dt  # ONE DAY

    temp_data = torch.tensor(pd.read_csv(csv_path, skiprows=0).iloc[:, 2:].to_numpy(dtype=np.float32),
                             dtype=torch.float32)
    time_data = torch.tensor(pd.read_csv(csv_path, skiprows=0).iloc[:, 1], dtype=torch.float64)

    Tin_continuous = Interp1D(time_data, temp_data[:, 0:len(model.building.rooms)].squeeze().T, method='linear')

    # Transform parameters
    if model.transform:
        theta = model.transform(model.params)
        model.cool_load = model.transform(model.loads)  # Watts for each room
    else:
        theta = model.params
        model.cool_load = model.loads

    # Scale inputs up to their physical values
    theta = model.scaling.physical_param_scaling(theta)
    model.cool_load = model.scaling.physical_cooling_scaling(model.cool_load)

    model.building.update_inputs(theta)

    latent = LatentSim(model)

    lumped = torch.tensor([])
    out = torch.tensor([])
    ind = torch.tensor([])

    n = 10
    for i in trange(n):
        pred = latent(time_data, Tin_continuous)
        model.iv = pred[-1]
        pred = pred.squeeze(-1)

        lumped = torch.concat((lumped, pred))
        out = torch.concat((out, model.Tout_continuous(time_data)))
        ind = torch.concat((ind, temp_data[:, 0:len(model.building.rooms)].squeeze().T))

    t = torch.flatten(torch.stack([time_data + i * ((time_data[-1]-time_data[0])+(time_data[1]-time_data[0])) for i in range(n)]))
    t = (t - t[0]) / (24*60**2)

    plt.plot(t.detach().numpy(), lumped.detach().numpy())
    plt.plot(t.detach().numpy(), out.detach().numpy())
    plt.plot(t.detach().numpy(), ind.detach().numpy())
    plt.show()


if __name__ == '__main__':

    with torch.no_grad():
        worker(0)


