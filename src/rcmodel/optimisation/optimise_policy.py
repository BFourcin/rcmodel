from ray.rllib.algorithms.algorithm import Algorithm
from rcmodel.optimisation.optimise_rc import test
from ray.rllib.models.preprocessors import NoPreprocessor, get_preprocessor


results = algo_ppo.train()


class OptimisePolicy:
    """
    Parameters
    ----------
    model : object
        RCModel class object.
    csv_path : string
        Path to .csv file containing room temperature data.
        Data will be sorted if not done already and saved to a new file with the tag '_sorted'
    sample_size : int
        Length of indexes to sample from dataset per batch.
    dt : int
        Timestep data will be resampled to.
    lr : float
        Learning rate for optimiser.
    model_id : int
        Unique identifier used when optimising multiple models.

    see https://docs.ray.io/en/latest/using-ray-with-pytorch.html
    """

    def __init__(self, env_config, ppo_config, policy_weights, train_dataset, test_dataset, opt_id=0):

        self.env = tools.env_creator(env_config)
        self.rl_algorithm = ppo_config.build().set_weights(policy_weights)

        # THIS isnt sorted below:
        self.model_id = opt_id
        self.train_dataset = train_dataset
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=1, shuffle=False
        )
        self.test_dataset = test_dataset
        self.test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=1, shuffle=False
        )


        # Check that we don't need preprocessing.
        prep = get_preprocessor(env.observation_space)
        assert prep.__name__ == 'NoPreprocessor', 'Not implemented preprocessing.'

    @staticmethod
    def train():
        results = rl_algorithm.train()
        avg_reward = results["episode_reward_mean"]
        return avg_reward

    def test(self):
        self.env.dataloader = self.test_dataloader
        avg_reward = test(self.env, self.rl_algorithm)
        return avg_reward



