
import numpy as np
from PIL import Image
import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset

from platforms.platform import get_platform

class ShuffleSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, num_examples):
        self._num_examples = num_examples
        self._seed = -1

    def __iter__(self):
        if self._seed == -1:
            indices = list(range(self._num_examples))
        elif self._seed is None:
            indices = torch.randperm(self._num_examples).tolist()
        else:
            g = torch.Generator()
            if self._seed is not None: g.manual_seed(self._seed)
            indices = torch.randperm(self._num_examples, generator=g).tolist()

        return iter(indices)

    def __len__(self):
        return self._num_examples

    def shuffle_dataorder(self, seed: int):
        self._seed = seed


class DistributedShuffleSampler(torch.utils.data.distributed.DistributedSampler):
    def __init__(self, dataset):
        super(DistributedShuffleSampler, self).__init__(
            dataset, num_replicas=get_platform().world_size, rank=get_platform().rank)
        self._seed = -1

    def __iter__(self):
        indices = torch.arange(len(self.dataset))

        if self._seed != -1:
            g = torch.Generator()
            g.manual_seed(self._seed or np.random.randint(10e6))
            perm = torch.randperm(len(indices), generator=g)
            indices = indices[perm]

        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices.tolist())

    def shuffle_dataorder(self, seed: int):
        self._seed = seed


class DataLoader(torch_geometric.loader.DataLoader):
    """A wrapper that makes it possible to access the custom shuffling logic."""

    def __init__(self, dataset: InMemoryDataset, batch_size: int, num_workers: int, pin_memory: bool = True):
        if get_platform().is_distributed:
            self._sampler = DistributedShuffleSampler(dataset)
        else:
            self._sampler = ShuffleSampler(len(dataset))

        self._iterations_per_epoch = np.ceil(len(dataset) / batch_size).astype(int)
        if get_platform().is_distributed:
            batch_size //= get_platform().world_size
            num_workers //= get_platform().world_size

        super(DataLoader, self).__init__(
            dataset, batch_size, sampler=self._sampler, num_workers=num_workers,
            pin_memory=pin_memory and get_platform().torch_device.type == 'cuda')

    def shuffle(self, seed: int):
        self._sampler.shuffle_dataorder(seed)

    @property
    def iterations_per_epoch(self):
        return self._iterations_per_epoch
