import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm, trange

from rcmodel import *
from main import initialise_model, RayActor

opt_id = 0
weather_data_path = '/home/benf/LSI/Data/Met Office Weather Files/JuneSept.csv'
csv_path = '/home/benf/LSI/Data/DummyData/train5wk_sorted.csv'  # where building data
dir_path = f'./outputs/run{opt_id}/models/'  # where to save
# Load model?
load_model_path = 'rcmodel_start.pt'  # or None


def do_plots():
    fig = plt.figure(figsize=(10, 7), dpi=400)
    plt.plot(range(1, len(avg_train_loss_plot) + 1), avg_train_loss_plot, label='train loss')
    plt.plot(range(1, len(avg_test_loss_plot) + 1), avg_test_loss_plot, label='test loss')
    plt.title('Train and Test Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    fig.savefig(f'./outputs/run{opt_id}/plots/LossPlot.png')
    plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(10, 7), dpi=400)
    y = torch.flatten(torch.tensor(ER_plot)).detach().numpy()
    axs[0].plot(range(1, len(y) + 1), y, 'b', label='expected rewards')
    axs[0].legend()

    y = torch.flatten(torch.tensor(rewards_plot)).detach().numpy()
    axs[1].plot(range(1, len(y) + 1), y, 'r', label='total rewards')
    axs[1].legend()
    fig.suptitle('Rewards', fontsize=16)
    axs[0].set_xlabel('Epoch')
    axs[1].set_xlabel('Epoch')
    axs[0].set_ylabel('Reward')
    axs[1].set_ylabel('Reward')

    fig.savefig(f'./outputs/run{opt_id}/plots/RewardsPlot.png')
    plt.close()


def init_scaling():
    # Initialise scaling class
    rm_CA = [100, 1e4]  # [min, max] Capacitance/area
    ex_C = [1e3, 1e8]  # Capacitance
    R = [0.1, 5]  # Resistance ((K.m^2)/W)
    scaling = InputScaling(rm_CA, ex_C, R)
    return scaling


# Initialise Model
scaling = init_scaling()
policy = PolicyNetwork(7, 2)
model = initialise_model(policy, scaling, weather_data_path)

model.save(0, dir_path)  # save initial model

# Initialise Optimise Class - for training physical model
dt = 30
sample_size = 7 * 24 * 60 ** 2 / dt  # ONE WEEK
op = OptimiseRC(model, csv_path, sample_size, dt, lr=1e-3, opt_id=opt_id)

# Initialise Reinforcement Class - for training policy
time_data = torch.tensor(pd.read_csv(csv_path, skiprows=0).iloc[:, 1], dtype=torch.float64)
temp_data = torch.tensor(pd.read_csv(csv_path, skiprows=0).iloc[:, 2:].to_numpy(dtype=np.float32), dtype=torch.float32)
env = LSIEnv(model, time_data)
rl = Reinforce(env, time_data, temp_data, alpha=1e-2)

# lists to keep track of process, used for plot at the end
avg_train_loss_plot = []
avg_test_loss_plot = []
rewards_plot = []
ER_plot = []
count = 1  # Counts number of epochs since start.

# Dataloader to be used when plotting a model run. This is just for info.
plot_results_data = tools.BuildingTemperatureDataset(csv_path, 5 * sample_size, all=True)
plot_dataloader = torch.utils.data.DataLoader(plot_results_data, batch_size=1, shuffle=False)

# check if dir exists and make if needed
Path(f'./outputs/run{opt_id}/plots/results/').mkdir(parents=True, exist_ok=True)


if load_model_path:
    model.load(load_model_path)


start_num = 0  # Number of cycles to start at. Used if resuming the run. i.e. the first cycle is (start_num + 1)

# Convergence happens when the mean gradient of loss/reward is < tol.
tol_phys = 5e-3
tol_policy = 50


cycles = 200
max_epochs = 200

plt.ioff()  # Reduces memory usage by matplotlib
for cycle in trange(cycles):

    # -------- Physical model training --------
    tqdm.write('Physical Model Training:')
    diff = torch.ones(5) * 10
    loss_prev = 0
    for epoch in range(max_epochs):
        avg_train_loss = op.train()
        avg_train_loss_plot.append(avg_train_loss)

        # Get difference from previous
        indx = epoch % len(diff)  # cycles the indexes in array
        diff[indx] = abs(loss_prev - avg_train_loss)  # keeps track of a window of differences
        loss_prev = avg_train_loss

        # Test Loss
        avg_test_loss = op.test()
        avg_test_loss_plot.append(avg_test_loss)

        tqdm.write(
            f'Epoch {count}, Train/Test Loss: {avg_train_loss:.2f}/{avg_test_loss:.2f}, Mean diff: {diff.mean():.3f}')

        # Save Model
        model_id = count + start_num
        model.save(model_id, dir_path)
        count += 1

        # check if mean of difference is less than tol. i.e. convergence
        if diff.mean() < tol_phys:
            break

    tqdm.write(f'Physical converged in {epoch + 1} epochs. Total epochs: {count - 1}\n')

    # Save a plot of results after physical training
    pltsolution_1rm(model, plot_dataloader,
                    f'./outputs/run{opt_id}/plots/results/Result_Cycle{start_num + cycle + 1}a.png')

    # -------- Policy training --------
    tqdm.write('Policy Training:')
    diff = torch.ones(5) * 1000
    rewards_prev = 0
    for epoch in range(max_epochs):
        rewards, ER = rl.train(1, sample_size)

        indx = epoch % len(diff)
        diff[indx] = abs(rewards_prev - torch.tensor(rewards))  # keeps track of a window of differences
        rewards_prev = torch.tensor(rewards)
        rewards_plot.append(rewards)
        ER_plot.append(ER)

        tqdm.write(
            f'Epoch {count}, Rewards/Expected Rewards: {torch.tensor(rewards).sum().item():.2f}/{torch.tensor(ER).sum().item():.2f}, Mean diff: {torch.mean(diff):.2f}')

        # Save Model
        model_id = count + start_num
        model.save(model_id, dir_path)
        count += 1

        if torch.mean(diff) < tol_policy:
            break

    tqdm.write(f'Policy converged in {epoch + 1} epochs. Total epochs: {count - 1}\n')

    # Save a plot of results after policy training
    pltsolution_1rm(model, plot_dataloader, f'./outputs/run{opt_id}/plots/results/Result_Cycle{start_num + cycle + 1}b.png')

    # Plot loss and reward plot
    do_plots()


