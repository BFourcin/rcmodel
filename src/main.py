import time
import torch
import pandas as pd
import numpy as np
import ray
from matplotlib import pyplot as plt
from tqdm.auto import tqdm, trange
from xitorch.interpolate import Interp1D
from pathlib import Path

from rcmodel import *
from rcmodel.optimisation import dataset_creator

from rcmodel.reinforce import PolicyNetwork
from rcmodel.reinforce import Reinforce
from rcmodel.reinforce import LSIEnv


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


class RayActor:
    def __init__(self, scaling, weather_data_path, csv_path, ):
        self.scaling = scaling
        self.weather_data_path = weather_data_path
        self.csv_path = csv_path

    def worker(self, opt_id, epochs=100):
        torch.set_num_threads(1)
        # csv_path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/DummyData/train5d_sorted.csv'
        # csv_path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/DummyData/test2d_sorted.csv'
        # csv_path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/210813data_sorted.csv'

        dir_path = f'./outputs/run{opt_id}/models/'  # where to save

        policy = PolicyNetwork(7, 2)
        model = initialise_model(policy, self.scaling, self.weather_data_path)

        model.save(0, dir_path)  # save initial model

        dt = 30
        sample_size = 24*60**2/dt

        op = OptimiseRC(model, self.csv_path, sample_size, dt, lr=1e-3, opt_id=opt_id)

        time_data = torch.tensor(pd.read_csv(self.csv_path, skiprows=0).iloc[:, 1], dtype=torch.float64)
        temp_data = torch.tensor(pd.read_csv(self.csv_path, skiprows=0).iloc[:, 2:].to_numpy(dtype=np.float32), dtype=torch.float32)

        env = LSIEnv(model, time_data)
        do_reinforce = Reinforce(env, time_data, temp_data, alpha=1e-2)

        avg_train_loss_plot = []
        avg_test_loss_plot = []
        rewards_plot = []
        ER_plot = []

        plot_results_data = tools.BuildingTemperatureDataset(self.csv_path, 5*sample_size, all=True)
        plot_dataloader = torch.utils.data.DataLoader(plot_results_data, batch_size=1, shuffle=False)

        # check if dir exists and make if needed
        Path(f'./outputs/run{opt_id}/plots/results/').mkdir(parents=True, exist_ok=True)

        for epoch in trange(epochs):
            # Physical Model Training:
            avg_train_loss = op.train()
            avg_train_loss_plot.append(avg_train_loss)

            # Policy Training:
            num_episodes = 1
            step_size = sample_size
            rewards, ER = do_reinforce.train(num_episodes, step_size)
            rewards_plot.append(rewards)
            ER_plot.append(ER)

            # Test Loss
            avg_test_loss = op.test()
            avg_test_loss_plot.append(avg_test_loss)

            # Save a plot of results every 10 epochs or if first/final epoch
            if (epoch+1) % 10 == 0 or epoch == epochs-1 or epoch == 0:
                pltsolution_1rm(model, plot_dataloader, f'./outputs/run{opt_id}/plots/results/Result_Epoch{epoch+1}.png')

            # Save Model
            model_id = epoch+1
            model.save(model_id, dir_path)

            tqdm.write(f'Epoch {epoch+1}, Train/Test Loss: {avg_train_loss:.2f}/{avg_test_loss:.2f}, Policy Rewards {sum(rewards).item():.1f}, Policy Expected Rewards {sum(ER).item():.1f}')

        final_train_loss = avg_train_loss
        final_test_loss = avg_test_loss

        _ = plt.figure(figsize=(10, 7), dpi=400)
        plt.plot(range(1, epochs+1), avg_train_loss_plot, label='train loss')
        plt.plot(range(1, epochs+1), avg_test_loss_plot, label='test loss')
        plt.title('Train and Test Loss During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Squared Error')
        plt.legend()
        plt.savefig(f'./outputs/run{opt_id}/plots/LossPlot.png')
        plt.close()

        fig, axs = plt.subplots(1, 2, figsize=(10, 7), dpi=400)
        axs[0].plot(range(1, epochs+1), torch.flatten(torch.tensor(ER_plot)).detach().numpy(), 'b', label='expected rewards')
        axs[0].legend()

        axs[1].plot(range(1, epochs+1), torch.flatten(torch.tensor(rewards_plot)).detach().numpy(), 'r', label='total rewards')
        axs[1].legend()
        fig.suptitle('Rewards', fontsize=16)
        axs[0].set_xlabel('Epoch')
        axs[1].set_xlabel('Epoch')
        axs[0].set_ylabel('Reward')
        axs[1].set_ylabel('Reward')

        plt.savefig(f'./outputs/run{opt_id}/plots/RewardsPlot.png')
        plt.close()

        final_params = model.transform(model.params).detach().numpy()
        final_cooling = model.transform(model.cooling).detach().numpy()

        return np.concatenate(([opt_id], final_params, final_cooling, [final_train_loss, final_test_loss]))

    def worker_no_cooling(self, opt_id, epochs=100):
        torch.set_num_threads(1)

        # csv_path = '../../Data/DummyData/train5d_sorted.csv'
        # csv_path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/DummyData/test2d_sorted.csv'

        dir_path = f'./outputs/run{opt_id}/models/'  # where to save models

        policy = None
        model = initialise_model(policy, self.scaling, self.weather_data_path)
        model.save(0, dir_path)  # save initial model

        dt = 30
        sample_size = 24 * 60 ** 2 / dt

        op = OptimiseRC(model, self.csv_path, sample_size, dt, lr=1e-2)

        avg_train_loss_plot = []
        avg_test_loss_plot = []

        plot_results_data = tools.BuildingTemperatureDataset(self.csv_path, 5 * sample_size, all=True)
        plot_dataloader = torch.utils.data.DataLoader(plot_results_data, batch_size=1, shuffle=False)

        # check if dir exists and make if needed
        Path(f'./outputs/run{opt_id}/plots/results/').mkdir(parents=True, exist_ok=True)

        for epoch in trange(epochs):
            # Physical Model Training:
            avg_train_loss = op.train()
            avg_train_loss_plot.append(avg_train_loss)

            # Test Loss
            avg_test_loss = op.test()
            avg_test_loss_plot.append(avg_test_loss)

            # Save a plot of results every 10 epochs or if first/final epoch
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1 or epoch == 0:
                pltsolution_1rm(model, plot_dataloader,
                                f'./outputs/run{opt_id}/plots/results/Result_Epoch{epoch + 1}.png')

            # Save Model
            model_id = epoch + 1
            model.save(model_id, dir_path)

            tqdm.write(f'Epoch {epoch + 1}, Train/Test Loss: {avg_train_loss:.2f}/{avg_test_loss:.2f}')

        final_train_loss = avg_train_loss
        final_test_loss = avg_test_loss

        _ = plt.figure(figsize=(10, 7), dpi=400)
        plt.plot(range(1, epochs + 1), avg_train_loss_plot, label='train loss')
        plt.plot(range(1, epochs + 1), avg_test_loss_plot, label='test loss')
        plt.title('Train and Test Loss During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Squared Error')
        plt.legend()
        plt.savefig(f'./outputs/run{opt_id}/plots/LossPlot.png')
        plt.close()

        final_params = model.transform(model.params).detach().numpy()
        final_cooling = model.transform(model.cooling).detach().numpy()

        return np.concatenate(([opt_id], final_params, final_cooling, [final_train_loss, final_test_loss]))


if __name__ == '__main__':
    use_ray = False

    if use_ray:
        num_cpus = 4
        num_jobs = num_cpus
        epochs = 2

        ray.init(num_cpus=num_cpus)

        # Initialise scaling class
        rm_CA = [100, 1e4]  # [min, max] Capacitance/area
        ex_C = [1e3, 1e8]  # Capacitance
        R = [0.1, 5]  # Resistance ((K.m^2)/W)
        Q_limit = [-10000, 10000]  # Cooling limit in W and gain limit in W/m2
        scaling = InputScaling(rm_CA, ex_C, R, Q_limit)
        weather_data_path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/Met Office Weather Files/JuneSept.csv'
        csv_path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/DummyData/test2d_sorted.csv'

        RayActor = ray.remote(RayActor)

        actors = [RayActor.remote(scaling, weather_data_path, csv_path) for _ in range(num_jobs)]

        results = ray.get([a.worker.remote(num, epochs) for num, a in enumerate(actors)])
        # results = ray.get([a.worker_no_cooling.remote(num, epochs) for num, a in enumerate(actors)])

        time.sleep(3)  # Print is cut short without sleep

        ray.shutdown()

        params_heading = ['Rm Cap/m2 (J/K.m2)', 'Ext Wl Cap 1 (J/K)', 'Ext Wl Cap 2 (J/K)', 'Ext Wl Res 1 (K.m2/W)', 'Ext Wl Res 2 (K.m2/W)', 'Ext Wl Res 3 (K.m2/W)', 'Int Wl Res (K.m2/W)']
        cooling_heading = ['Cooling (W)']
        headings = [['Run Number'], params_heading, cooling_heading, ['Final Average Train Loss', 'Final Avg Test Loss']]
        flat_list = [item for sublist in headings for item in sublist]

        df = pd.DataFrame(np.array(results), columns=flat_list)
        df.to_csv('./outputs/results.csv', index=False,)

    else:
        # Initialise scaling class
        rm_CA = [100, 1e4]  # [min, max] Capacitance/area
        ex_C = [1e3, 1e8]  # Capacitance
        R = [0.1, 5]  # Resistance ((K.m^2)/W)
        Q_limit = [-10000, 10000]  # Cooling limit in W and gain limit in W/m2
        scaling = InputScaling(rm_CA, ex_C, R, Q_limit)
        weather_data_path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/Met Office Weather Files/JuneSept.csv'
        csv_path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/DummyData/test2d_sorted.csv'

        actor = RayActor(scaling, weather_data_path, csv_path)

        results = actor.worker(0, 5)

        params_heading = ['Rm Cap/m2 (J/K.m2)', 'Ext Wl Cap 1 (J/K)', 'Ext Wl Cap 2 (J/K)', 'Ext Wl Res 1 (K.m2/W)',
                          'Ext Wl Res 2 (K.m2/W)', 'Ext Wl Res 3 (K.m2/W)', 'Int Wl Res (K.m2/W)', 'Offset Gain (W/m2)']
        cooling_heading = ['Cooling (W)']
        headings = [['Run Number'], params_heading, cooling_heading,
                    ['Final Average Train Loss', 'Final Avg Test Loss']]
        flat_list = [item for sublist in headings for item in sublist]

        df = pd.DataFrame(np.array([results]), columns=flat_list)
        df.to_csv('./outputs/results.csv', index=False, )


