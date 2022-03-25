import torch
import pandas as pd
from ..tools.helper_functions import dataset_creator


def train(model, dataloader, optimizer):
    """
    Performs one epoch of training.
    Order of rooms in building and in data must match otherwise model will fit wrong rooms to data.
    """
    model.reset_iv()  # Reset initial value
    model.train()

    # if policy exists:
    if model.cooling_policy:
        model.cooling_policy.eval()

        # This ended up causing the policy not to optimise:
        # Stops Autograd endlessly keeping track of the graph. Memory Leak!
        # try:
        #     for layer in model.cooling_policy.parameters():
        #         layer.requires_grad = False
        #
        # # This occurs when policy contains no parameters e.g. when using prior to train.
        # except AttributeError:
        #     pass

    num_cols = len(model.building.rooms)  # number of columns to use from data.
    num_batches = len(dataloader)
    train_loss = 0
    loss_fn = torch.nn.MSELoss()

    for batch, (time, temp) in enumerate(dataloader):
        # Get model arguments:
        time = time.squeeze(0)
        temp = temp.squeeze(0)

        # Compute prediction and loss
        pred = model(time)
        pred = pred.squeeze(-1)  # change from column to row matrix

        loss = loss_fn(pred[:, 2:], temp[:, 0:num_cols])
        train_loss += loss.item()

        # get last output and use for next initial value
        model.iv = pred[-1, :].unsqueeze(1).detach()  # MUST DETACH GRAD

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return train_loss / num_batches


def test(model, dataloader):
    model.reset_iv()  # Reset initial value
    model.eval()  # Put model in evaluation mode
    num_batches = len(dataloader)
    num_cols = len(model.building.rooms)  # number of columns to take from data.
    test_loss = 0
    loss_fn = torch.nn.MSELoss()

    with torch.no_grad():
        for (time, temp) in dataloader:
            time = time.squeeze(0)
            temp = temp.squeeze(0)

            pred = model(time)
            pred = pred.squeeze(-1)  # change from column to row matrix
            test_loss += loss_fn(pred[:, 2:], temp[:, 0:num_cols]).item()

            # get last output and use for next initial value
            model.iv = pred[-1, :].unsqueeze(1).detach()  # MUST DETACH GRAD

    test_loss /= num_batches

    return test_loss


class OptimiseRC:
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

    def __init__(self, model, csv_path, sample_size, dt=30, lr=1e-3, opt_id=0):
        self.model = model
        # self.model.init_physical()  # randomise parameters - DO WE WANT THIS?
        self.model_id = opt_id
        self.train_dataloader, self.test_dataloader = dataset_creator(csv_path, int(sample_size), int(dt))

        self.optimizer = torch.optim.Adam([self.model.params, self.model.loads], lr=lr)

    def train(self):
        avg_loss = train(self.model, self.train_dataloader, self.optimizer)
        return avg_loss

    def test(self):
        test_loss = test(self.model, self.test_dataloader)
        return test_loss

    def train_loop(self, epochs):
        print(self.model.params)
        for i in range(int(epochs)):
            # print(f"Epoch {i + 1}\n-------------------------------")
            testloss = self.train()

        results = [testloss, self.model]
        return results
