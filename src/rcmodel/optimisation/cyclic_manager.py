import dill
import csv
import numpy as np
from pathlib import Path
from PIL import Image
from ray.rllib.algorithms.algorithm import Algorithm
from rcmodel.optimisation.optimise_models import test
from tqdm import tqdm, trange
from datetime import datetime


def capped_cubic_test_schedule(episode_id: int) -> bool:
    if episode_id < 1000:
        return int(round(episode_id ** (1.0 / 3))) ** 3 == episode_id
    else:
        return episode_id % 1000 == 0


class OptimiseManager:
    def __init__(self, physical_optimiser, policy_optimiser, physical_loops=100,
                 policy_loops=100, render_phase=None, render_trigger=None,
                 physical_test_trigger=None, policy_test_trigger=None, logging=True,
                 log_filename='log.csv', verbose=True):

        self.physical_optimiser = physical_optimiser
        self.policy_optimiser = policy_optimiser

        if render_phase:
            assert render_phase in {'train', 'test', 'both'}

        self.render_phase = render_phase

        self.cycle_count = 0
        self.physical_epochs = 0
        self.policy_epochs = 0

        self.physical_loops = physical_loops
        self.policy_loops = policy_loops

        if physical_test_trigger is None:
            def physical_test_trigger(x): return True

        if policy_test_trigger is None:
            def policy_test_trigger(x): return True

        self.physical_test_trigger = physical_test_trigger
        self.policy_test_trigger = policy_test_trigger

        if render_trigger is None:
            def render_trigger(x): return True

        self.render_trigger = render_trigger

        # get datetime so we can save images in seperate directories.
        now = datetime.now()
        self.directory_path = "./outputs/" + now.strftime("%y%m%d_%H%M")

        # Logging
        self.logging = logging
        self.log_filename = self.directory_path + '/' + log_filename

        self.verbose = verbose

        print(f'Run outputs saved to: {Path(self.directory_path).resolve()}')

    def cycle(self):

        # ------ PHYSICAL ------
        for epoch in trange(self.physical_loops, desc='Physical Training'):
            self.running_physical = True  # Just some flags.
            self.running_policy = False

            # ------ Train ------
            # Run the physical optimiser.
            reward_list, render_list = self.train_physical()
            self.physical_epochs += 1

            # If images have been rendered, save them.
            if render_list:
                directory_path = self.directory_path + "/images/train"
                self.save_image_list(render_list, directory_path)

            # ------ Test ------
            # Run test if required:
            if self.physical_test_trigger(self.physical_epochs):
                reward_list_test, render_list = self.test()

                if render_list:
                    directory_path = self.directory_path + "/images/test"
                    self.save_image_list(render_list, directory_path)

            info = {
                'cycle': self.cycle_count,
                'epoch': self.physical_epochs + self.policy_epochs,
                'average train reward': np.mean(reward_list),
                'average test reward': np.mean(reward_list_test),
                'physical_epoch': self.running_physical,
                'policy_epoch': self.running_policy}

            if self.verbose:
                print_training_info(info)

            if self.logging:
                self.log_data_to_csv(info)

        # ------ Save ------
        self.save()

        # ------ Handover ------
        self.handover_physical_to_policy()

        # ------ POLICY ------
        for epoch in trange(self.policy_loops, desc='Policy Training'):
            self.running_physical = False  # Just some flags.
            self.running_policy = True

            # ------ Train ------
            # Run the policy optimiser.
            avg_reward = self.train_policy()
            self.policy_epochs += 1

            # ------ Test ------
            if self.policy_test_trigger(self.policy_epochs):
                reward_list_test, render_list = self.test()

                if render_list:
                    directory_path = self.directory_path + "/images/test"
                    self.save_image_list(render_list, directory_path)

            info = {
                'cycle': self.cycle_count,
                'epoch': self.physical_epochs + self.policy_epochs,
                'average train reward': avg_reward,
                'average test reward': np.mean(reward_list_test),
                'physical_epoch': self.running_physical,
                'policy_epoch': self.running_policy}

            if self.verbose:
                print_training_info(info)

            if self.logging:
                self.log_data_to_csv(info)

        # ------ Save ------
        self.save()

        # ------ Handover ------
        self.handover_policy_to_physical()

        # ------ Sanity Checks ------
        self.sanity_checks()

        self.cycle_count += 1

    def train_physical(self):
        # See if we should render:
        if self.render_phase == 'train' or self.render_phase == 'both':
            start_render = self.render_trigger(self.physical_epochs)
        else:
            start_render = False

        self.physical_optimiser.env.unwrapped.recording = start_render

        # Train the physical model for one epoch.
        return self.physical_optimiser.train()

    def train_policy(self):
        # We cant render training of the policy.

        # Train the policy model for one epoch.
        return self.policy_optimiser.train()

    def test(self):
        # See if we should render:
        if self.render_phase == 'test' or self.render_phase == 'both':

            start_render = self.render_trigger(self.physical_epochs)
        else:
            start_render = False

        self.physical_optimiser.env.unwrapped.recording = start_render

        env = self.physical_optimiser.env
        rl_algorithm = self.policy_optimiser.rl_algorithm  # rl_alg can come from either
        dataloader = self.physical_optimiser.test_dataloader

        reward_list, render_list = test(env, rl_algorithm, dataloader)

        return reward_list, render_list

    def save(self, directory_path=None, filename=None):

        if directory_path is None:
            directory_path = self.directory_path + "/models"
        if filename is None:
            filename = f"model_{self.physical_epochs}_{self.policy_epochs}.pkl"

        directory_path = Path(directory_path)
        directory_path.mkdir(parents=True, exist_ok=True)

        checkpoint_path = self.policy_optimiser.rl_algorithm.save(directory_path)
        rl_algorithm = self.policy_optimiser.rl_algorithm

        self.physical_optimiser.rl_algorithm = None
        self.policy_optimiser.rl_algorithm = None

        save_dict = {
            "checkpoint_path": checkpoint_path,
            "manager": self,
        }

        filepath = directory_path / filename
        with open(filepath, "wb") as dill_file:
            dill.dump(save_dict, dill_file)

        # Put back
        self.physical_optimiser.rl_algorithm = rl_algorithm
        self.policy_optimiser.rl_algorithm = rl_algorithm

        return filename

    @staticmethod
    def load(filename):
        """
        Parameters
        ----------
        filename : str
            Path to the pickled object to load.
        Returns
        ----------
        RCModel
        """
        from ray.tune.registry import register_env
        from rcmodel import env_creator

        register_env("LSIEnv", env_creator)

        with open(filename, "rb") as dill_file:

            save_dict = dill.load(dill_file)

            manager = save_dict["manager"]
            # Put back the batch generator
            manager.physical_optimiser.env.unwrapped.batch_generator =\
                iter(manager.physical_optimiser.env.unwrapped.dataloader)

            # Get the RL algorithm from the checkpoint and put back.
            checkpoint_path = save_dict["checkpoint_path"]
            rl_algorithm = Algorithm.from_checkpoint(checkpoint_path)
            manager.physical_optimiser.rl_algorithm = rl_algorithm
            manager.policy_optimiser.rl_algorithm = rl_algorithm

            return manager

    def handover_physical_to_policy(self):
        env_changes = {
            "update_state_dict": self.physical_optimiser.env.RC.state_dict(),
            "dataloader": self.physical_optimiser.train_dataloader,
        }

        # This also recalculates the latent variables.
        self.policy_optimiser.update_environment(env_changes)

    def handover_policy_to_physical(self):
        # We might not need to do anything here!
        pass

    def sanity_checks(self):
        assert self.policy_optimiser.rl_algorithm is self.physical_optimiser.rl_algorithm, "We're not using the same RL algorithm!"

    def save_image_list(self, image_list, directory_path):

        if not isinstance(image_list, list):
            image_list = [image_list]

        Path(directory_path).mkdir(parents=True, exist_ok=True)
        for i, image in enumerate(image_list):
            # Convert the numpy array to a PIL Image object
            pil_image = Image.fromarray(image)

            # Save the image to disk using the specified filename and format
            filename = f"{directory_path}/img{self.physical_epochs}_{self.policy_epochs}_{i+1}.png"
            pil_image.save(filename)

    def log_data_to_csv(self, data):
        # Extract directory path from the filename
        directory = Path(self.log_filename).parent

        # Create the directory if it doesn't exist
        directory.mkdir(parents=True, exist_ok=True)

        # Check if the CSV file already exists
        file_exists = Path(self.log_filename).is_file()

        with open(self.log_filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=data.keys())

            # Write header row if the file doesn't exist
            if not file_exists:
                writer.writeheader()

            # Write data row
            writer.writerow(data)


def print_training_info(info):
    cycle = f"{info['cycle']:>5}"
    epoch = f"{info['epoch']:>5}"
    avg_train_reward = f"{info['average train reward']:.2f}"
    avg_test_reward = f"{info['average test reward']:.2f}"
    physical_epoch = 'Y' if info['physical_epoch'] else 'N'
    policy_epoch = 'Y' if info['policy_epoch'] else 'N'

    if (info['epoch'] - 1) % 20 == 0:
        separator = "+-------+-------+----------------+----------------+----------+---------+"
        headers =   "| Cycle | Epoch |    Avg Train   |    Avg Test    | Physical | Policy  |"

        print(separator)
        print(headers)
        print(separator)

    output = f"| {cycle} | {epoch} | {avg_train_reward:>14} | {avg_test_reward:>14} | {physical_epoch:^8} | {policy_epoch:^7} |"
    print(output)
