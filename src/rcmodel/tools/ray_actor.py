import torch
import pandas as pd
import numpy as np
import ray
from matplotlib import pyplot as plt
from tqdm.auto import tqdm, trange
from pathlib import Path

from ..tools import InputScaling, BuildingTemperatureDataset, pltsolution_1rm


class RayActor:
    """
    Class provides basic framework to call ray for multiprocessing.
    """
    def __init__(self, model, csv_path, physical_training=True, policy_training=True):
        self.model = model
        self.csv_path = csv_path
        self.physical_training = physical_training
        self.policy_training = policy_training

    def worker(self, opt_id, epochs=100):
        from ..optimisation.optimise_rc import OptimiseRC  # placing here avoids circular import

        torch.set_num_threads(1)

        dir_path = f'./outputs/run{opt_id}/models/'  # where to save

        self.model.save(0, dir_path)  # save initial model

        dt = 30
        sample_size = 24 * 60 ** 2 / dt

        op = OptimiseRC(self.model, self.csv_path, sample_size, dt, lr=1e-3, opt_id=opt_id)

        if self.policy_training:
            time_data = torch.tensor(pd.read_csv(self.csv_path, skiprows=0).iloc[:, 1], dtype=torch.float64)
            temp_data = torch.tensor(pd.read_csv(self.csv_path, skiprows=0).iloc[:, 2:].to_numpy(dtype=np.float32),
                                     dtype=torch.float32)

            env = LSIEnv(self.model, time_data)
            do_reinforce = Reinforce(env, time_data, temp_data, alpha=1e-2)

            rewards_plot = []
            ER_plot = []

        avg_train_loss_plot = []
        avg_test_loss_plot = []

        plot_results_data = BuildingTemperatureDataset(self.csv_path, 5 * sample_size, all=True)
        plot_dataloader = torch.utils.data.DataLoader(plot_results_data, batch_size=1, shuffle=False)

        # check if dir exists and make if needed
        Path(f'./outputs/run{opt_id}/plots/results/').mkdir(parents=True, exist_ok=True)

        for epoch in trange(epochs):

            if self.physical_training:
                # Physical Model Training:
                avg_train_loss = op.train()
                avg_train_loss_plot.append(avg_train_loss)

            if self.policy_training:
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
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1 or epoch == 0:
                pltsolution_1rm(self.model, plot_dataloader,
                                f'./outputs/run{opt_id}/plots/results/Result_Epoch{epoch + 1}.png')

            # Save Model
            model_id = epoch + 1
            self.model.save(model_id, dir_path)

            if self.policy_training:
                tqdm.write(f'Epoch {epoch + 1}, Train/Test Loss: {avg_train_loss:.2f}/{avg_test_loss:.2f}, Policy '
                           f'Rewards {sum(rewards).item():.1f}, Policy Expected Rewards {sum(ER).item():.1f}')
            else:
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

        if self.policy_training:
            fig, axs = plt.subplots(1, 2, figsize=(10, 7), dpi=400)
            axs[0].plot(range(1, epochs + 1), torch.flatten(torch.tensor(ER_plot)).detach().numpy(), 'b',
                        label='expected rewards')
            axs[0].legend()

            axs[1].plot(range(1, epochs + 1), torch.flatten(torch.tensor(rewards_plot)).detach().numpy(), 'r',
                        label='total rewards')
            axs[1].legend()
            fig.suptitle('Rewards', fontsize=16)
            axs[0].set_xlabel('Epoch')
            axs[1].set_xlabel('Epoch')
            axs[0].set_ylabel('Reward')
            axs[1].set_ylabel('Reward')

            plt.savefig(f'./outputs/run{opt_id}/plots/RewardsPlot.png')
            plt.close()

        final_params = self.model.transform(self.model.params).detach().numpy()
        final_cooling = self.model.transform(self.model.cooling).detach().numpy()

        return np.concatenate(([opt_id], final_params, final_cooling, [final_train_loss, final_test_loss]))


if __name__ == '__main__':
    from rcmodel import *

    use_ray = True

    epochs = 2

    if use_ray:
        num_cpus = 4
        num_jobs = num_cpus
        ray.init(num_cpus=num_cpus)

    # Initialise scaling class
    rm_CA = [100, 1e4]  # [min, max] Capacitance/area
    ex_C = [1e3, 1e8]  # Capacitance
    R = [0.1, 5]  # Resistance ((K.m^2)/W)
    Q_limit = [-100, 100]  # Cooling limit and gain limit in W/m2
    scaling = InputScaling(rm_CA, ex_C, R, Q_limit)

    # Laptop:
    weather_data_path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/Met Office Weather ' \
                        'Files/JuneSept.csv'
    csv_path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/DummyData/test2d_sorted.csv'

    # Hydra:
    # weather_data_path = '/home/benf/LSI/Data/Met Office Weather Files/JuneSept.csv'
    # csv_path = '/home/benf/LSI/Data/DummyData/train5d_sorted.csv'

    prior = PriorCoolingPolicy()
    model = initialise_model(prior, scaling, weather_data_path)

    if use_ray:
        RayActor = ray.remote(RayActor)
        actors = [RayActor.remote(model, csv_path, policy_training=False) for _ in range(num_jobs)]

        results = ray.get([a.worker.remote(num, epochs) for num, a in enumerate(actors)])

        ray.shutdown()

    else:
        actor = RayActor(model, csv_path)
        actor.policy_training = False  # Turn cooling optimisation off
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

