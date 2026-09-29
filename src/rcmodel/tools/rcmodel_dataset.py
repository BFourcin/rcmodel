from random import randint

import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler


class BuildingTemperatureDataset(Dataset):
    """
    Splits dataset up into batches of len(dataset) // sample_size. Note remainder of data is thrown away.
    train and test tags can be used to select a percentage slice of data in this order,see function _split_dataset()
    for current % splits.

    If there is insufficient data for one batch, sample_size will be reduced to match the data.

    block_indices: optional list of window indices (block i is rows i*sample_size .. (i+1)*sample_size of
    the split) to walk instead of every window - how an interleaved evaluation split picks its windows.
    See interleaved_blocks().
    """

    def __init__(self, csv_path, sample_size, transform=None, all=True, train=False, test=False, block_indices=None):
        self.csv_path = csv_path
        self.transform = transform
        self.sample_size = int(sample_size)  # this is the number of rows of data from the csv per sample.
        self.headings = list(pd.read_csv(csv_path, nrows=1))  # list of dataframe headings
        self.all = all
        self.train = train
        self.test = test

        # auto splits data by train and test
        # entry count is number of rows to read from csv.
        # remainder of data e.g. after entry_count//sample_size is lost
        self.rows_to_skip, self.entry_count = self._split_dataset()

        self.block_indices = None if block_indices is None else [int(i) for i in block_indices]
        if self.block_indices is not None:
            n_blocks = self.entry_count // self.sample_size
            if not self.block_indices or min(self.block_indices) < 0 or max(self.block_indices) >= n_blocks:
                raise ValueError(f"block_indices must be within 0..{n_blocks - 1}, got {self.block_indices}.")

        self.__len__()  # Used to initialise logic used in __len__.

    def __len__(self):
        """
        Get number of batches in the dataset. Returns int
        Minimum of 1 batch will be returned.
        """
        if getattr(self, "block_indices", None) is not None:
            return len(self.block_indices)

        num_samples = self.entry_count // self.sample_size

        # Insufficient data for 1 whole sample size. Remainder of data used instead.
        if num_samples == 0:
            num_samples = 1
            self.sample_size = self.entry_count  # Reduce sample to the entries remaining

        return num_samples

    def __getitem__(self, idx):
        """
        Returns a 'sample' of the total dataset.

        Each sample has length 'sample_size' except for the final one, which contains the remainder of the dataset.

        The entire dataset is never loaded in at once, each sample is read in separately when needed to reduce memory usage.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if getattr(self, "block_indices", None) is not None:
            idx = self.block_indices[idx]

        # Get lower bound of dataset slice
        lb = idx * self.sample_size + self.rows_to_skip

        # Get pandas df of sample
        df_sample = pd.read_csv(self.csv_path, skiprows=lb, nrows=self.sample_size)

        # Get time column (time must be in the 1th column)
        t_sample = torch.tensor(df_sample.iloc[:, 1].values, dtype=torch.float64)  # units (s)

        # Get temp matrix
        temp_sample = torch.tensor(
            df_sample.iloc[:, 2:].values, dtype=torch.float32
        )  # pandas needs 2: to get all but first & second column

        # apply transforms if required
        if self.transform:
            temp_sample = self.transform(temp_sample)

        return t_sample, temp_sample

    def _get_entries(self):
        """Get total rows/entries of data in csv"""
        from csv import reader

        # the rows in the .csv are counted:
        with open(self.csv_path) as f:
            read_f = reader(f, delimiter=",")
            entry_count = sum(1 for row in read_f) - 1  # minus one to account for heading

        return entry_count

    def _split_dataset(self):

        train_split = 0.8
        test_split = 0.2

        total_entries = self._get_entries()  # total rows of data in csv

        if self.train:
            rows_to_skip = 0
            entry_count = int(total_entries * train_split)
            return rows_to_skip, entry_count
        elif self.test:
            rows_to_skip = int(total_entries * train_split)
            entry_count = int(total_entries * test_split)
            return rows_to_skip, entry_count
        elif self.all:
            rows_to_skip = 0
            entry_count = total_entries
            return rows_to_skip, entry_count
        else:
            raise ValueError("train, test and validation all False")

    def get_all_data(self):
        # Get upper and lower bounds of dataset slice
        lb = self.rows_to_skip

        # Get pandas df of sample
        df_sample = pd.read_csv(self.csv_path, skiprows=lb, nrows=self.entry_count)

        # Get time column (time must be in the 1th column)
        t_sample = torch.tensor(df_sample.iloc[:, 1].values, dtype=torch.float64)  # units (s)

        # Get temp matrix
        temp_sample = torch.tensor(
            df_sample.iloc[:, 2:].values, dtype=torch.float32
        )  # pandas needs 2: to get all but first & second column

        # apply transforms if required
        if self.transform:
            temp_sample = self.transform(temp_sample)

        return t_sample, temp_sample

    def get_history(self):
        """Every row from the START OF THE FILE to the end of this split, as (time, temperatures).

        What get_iv_array() warms the latent wall nodes up over. Those nodes are driven only by past
        outdoor and room temperatures, so running them over everything before the split is causal (no
        future data), and it means the split's first window starts from warmed-up wall states rather
        than from a steady-state guess made at the split's first row. For a test split that guess
        used to cost the first evaluation window more error than the rest of the split put together.
        With block_indices, the history runs to the end of the last block walked.
        """
        end = self.rows_to_skip + self.entry_count
        if getattr(self, "block_indices", None) is not None:
            end = self.rows_to_skip + (max(self.block_indices) + 1) * self.sample_size
        df = pd.read_csv(self.csv_path, nrows=end)
        t = torch.tensor(df.iloc[:, 1].values, dtype=torch.float64)
        temp = torch.tensor(df.iloc[:, 2:].values, dtype=torch.float32)
        if self.transform:
            temp = self.transform(temp)
        return t, temp


def interleaved_blocks(n_rows, sample_size, every):
    """Indices of the sample_size blocks an interleaved evaluation split holds out: every `every`-th
    block, starting with block every-1 (so the first held-out window has history before it).

    Spreading the held-out windows through the whole record, rather than taking its tail, means the
    selection metric sees every season and every operating regime - a tail split of a May-September
    cooling record is September, when the cooling barely runs.
    """
    n_blocks = n_rows // sample_size
    blocks = list(range(every - 1, n_blocks, every))
    if not blocks:
        raise ValueError(f"Only {n_blocks} blocks of {sample_size} rows: none to hold out every {every}.")
    return blocks


class RandomSampleDataset(Dataset):
    """
    Used to return random windows of data from a dataset.

    The initial part of the data (len=warmup_size) is reserved to be used to "warm-up" the latent variables of the
    model and is not returned by this class.

    Order of the data set goes: warm-up data, training data, testing data with no overlaps.
    len(dataset_train) == (len(data_set)-warmup_size) * 0.8
    len(dataset_test) == (len(data_set)-warmup_size) * 0.2
    """

    def __init__(
        self,
        csv_path,
        sample_size,
        warmup_size,
        transform=None,
        all=True,
        train=False,
        test=False,
        epoch_length=None,
        exclude_blocks=None,
    ):
        self.csv_path = csv_path
        self.transform = transform
        self.sample_size = int(sample_size)  # this is the number of rows of data from the csv per sample.
        self.warmup_size = int(warmup_size)  # amount of warmup data used to get a better result for iv.
        self.headings = list(pd.read_csv(csv_path, nrows=1))  # list of dataframe headings
        self.all = all
        self.train = train
        self.test = test

        # Force the size of each epoch e.g. epoch_length=1 means 1 batch (sample_size) of data for the whole epoch.
        # Useful to force quick cycles for testing and parameter searching.
        self.epoch_length = epoch_length

        # auto splits data by train and test
        # entry count is total number of rows in the csv which belong in this dataset.
        self.rows_to_skip, self.entry_count = self._split_dataset()

        # exclude_blocks: sample_size blocks (as interleaved_blocks() numbers them) held out for evaluation.
        # A window is only ever drawn from the gaps between them, so no training window overlaps one.
        self.exclude_blocks = sorted(int(b) for b in exclude_blocks) if exclude_blocks else []
        self._start_ranges = self._allowed_start_ranges()

        self.__len__()  # Used to initialise logic used in __len__.

    def _allowed_start_ranges(self):
        """[(first, last)] window start offsets (inclusive, relative to rows_to_skip) that avoid every
        excluded block."""
        ss = self.sample_size
        free, lo = [], 0
        for block in self.exclude_blocks:
            free.append((lo, block * ss))
            lo = (block + 1) * ss
        free.append((lo, self.entry_count))
        return [(a, b - ss) for a, b in free if b - a >= ss]

    def _random_start(self):
        ranges = getattr(self, "_start_ranges", None) or [(0, self.entry_count - self.sample_size)]
        counts = [last - first + 1 for first, last in ranges]
        pick = randint(0, sum(counts) - 1)
        for (first, _), count in zip(ranges, counts, strict=True):
            if pick < count:
                return first + pick
            pick -= count
        raise AssertionError("unreachable")

    def __len__(self):
        """
        Get number of batches in the dataset. Returns int
        Minimum of 1 batch will be returned.

        if epoch_length is set, the length is forced to this value.
        """
        if self.epoch_length:
            num_samples = self.epoch_length
        else:
            num_samples = self.entry_count // self.sample_size - len(getattr(self, "exclude_blocks", []))

            # Insufficient data for 1 whole sample size. Remainder of data used instead.
            if num_samples == 0:
                raise ValueError("Insufficient amount of data")

        return num_samples

    def __getitem__(self, idx=0):
        """
        Returns a 'sample' of the total dataset.

        A window of data is randomly selected from the data remaining after we have removed what was used in the warm up.

        The entire dataset is never loaded in at once, each sample is read in separately when needed to reduce memory usage.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # we take a random index from the range of valid indexes:
        start_idx = self._random_start() + self.rows_to_skip

        # Get pandas df of sample
        df_sample = pd.read_csv(self.csv_path, skiprows=start_idx, nrows=self.sample_size)

        # Get time column (time must be in the 1th column)
        t_sample = torch.tensor(df_sample.iloc[:, 1].values, dtype=torch.float64)  # units (s)

        # Get temp matrix
        temp_sample = torch.tensor(
            df_sample.iloc[:, 2:].values, dtype=torch.float32
        )  # pandas needs 2: to get all but first & second column

        # apply transforms if required
        if self.transform:
            temp_sample = self.transform(temp_sample)

        return t_sample, temp_sample

    def _get_entries(self):
        """Get total rows/entries of data in csv"""
        from csv import reader

        # the rows in the .csv are counted:
        with open(self.csv_path) as f:
            read_f = reader(f, delimiter=",")
            total_entries = sum(1 for row in read_f) - 1  # minus one to account for heading

        total_entries -= self.warmup_size  # remove the chunk of data used soley for warm up

        # check there is sufficient data
        assert total_entries > 0 and total_entries > self.sample_size

        return total_entries

    def _split_dataset(self):

        train_split = 0.8
        test_split = 0.2

        total_entries = self._get_entries()  # total rows of data in csv

        if self.train:
            rows_to_skip = self.warmup_size
            entry_count = int(total_entries * train_split)
            return rows_to_skip, entry_count
        elif self.test:
            rows_to_skip = int(total_entries * train_split) - self.warmup_size  # do warm up in training section.
            entry_count = int(total_entries * test_split)
            return rows_to_skip, entry_count
        elif self.all:
            rows_to_skip = self.warmup_size
            entry_count = total_entries
            return rows_to_skip, entry_count
        else:
            raise ValueError("train, test and validation all False")

    def get_all_data(self):
        start_idx = self.rows_to_skip

        # Get pandas df of entire valid dataset (iterate to avoid errors)
        # df_sample = pd.read_csv(self.csv_path, skiprows=start_idx, nrows=self.entry_count + self.sample_size)  #OLD
        iter_csv = pd.read_csv(
            self.csv_path, skiprows=start_idx, nrows=self.entry_count + self.sample_size, iterator=True, chunksize=10000
        )
        df_sample = pd.concat([chunk.dropna(how="all") for chunk in iter_csv])

        # Get time column (time must be in the 1th column)
        t_sample = torch.tensor(df_sample.iloc[:, 1].values, dtype=torch.float64)  # units (s)

        # Get temp matrix
        temp_sample = torch.tensor(df_sample.iloc[:, 2:].values, dtype=torch.float32)

        # apply transforms if required
        if self.transform:
            temp_sample = self.transform(temp_sample)

        return t_sample, temp_sample

    def get_history(self):
        """Every row from the START OF THE FILE to the end of this split, as (time, temperatures).

        What get_iv_array() warms the latent wall nodes up over. Those nodes are driven only by past
        outdoor and room temperatures, so running them over everything before the split is causal (no
        future data), and it means the split's first window starts from warmed-up wall states rather
        than from a steady-state guess made at the split's first row. For a test split that guess
        used to cost the first evaluation window more error than the rest of the split put together.
        """
        df = pd.read_csv(self.csv_path, nrows=self.rows_to_skip + self.entry_count)
        t = torch.tensor(df.iloc[:, 1].values, dtype=torch.float64)
        temp = torch.tensor(df.iloc[:, 2:].values, dtype=torch.float32)
        if self.transform:
            temp = self.transform(temp)
        return t, temp


class InfiniteSampler(Sampler):
    """Works with RandomSampleDataset to allow for infinite samples of data to be drawn.
    usage:
    train_loader = DataLoader(dataset_train, batch_size=batch_size, sampler=InfiniteSampler(dataset_train))"""

    def __init__(self, data_source):
        super().__init__(data_source)
        assert len(data_source) > 0
        self.dataset = data_source

    def __iter__(self):
        order = list(range(len(self.dataset)))
        idx = 0
        while True:
            yield order[idx]
            idx += 1
            if idx == len(order):
                idx = 0

    def __len__(self) -> int:
        return len(self.dataset)
