
import math
import torch
from torch.utils.data.sampler import Sampler
import torch.distributed as dist


class RepeatedDistSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        shuffle (optional): If true (default), sampler will shuffle the indices
    """

    def __init__(self, dataset, num_imgs, num_replicas=None, rank=None, shuffle=True, ba=False):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_imgs_rank = int(math.ceil(num_imgs * 1.0 / self.num_replicas))
        self.total_size = self.num_imgs_rank * self.num_replicas
        self.num_imgs = num_imgs
        self.shuffle = shuffle
        self.ba = ba


    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        n_repeats = self.num_imgs // len(self.dataset) + 1
        indices = []
        for n in range(n_repeats):
            if self.shuffle:
                g.manual_seed(n)
                indices += torch.randperm(len(self.dataset), generator=g).tolist()
            else:
                indices += [i for i in range(len(self.dataset))]

        # add extra samples to make it evenly divisible
        indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        if self.ba:
            n_rep = max(4, self.num_replicas)
            len_ind = len(indices) // n_rep + 1
            indices = indices[:len_ind]
            indices = [ind for ind in indices for _ in range(n_rep)]

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_imgs_rank

        return iter(indices)

    def __len__(self):
        return self.num_imgs_rank

