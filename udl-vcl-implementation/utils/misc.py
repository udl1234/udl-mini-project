import torch
from torch.distributions import kl_divergence, Normal
from torch.utils.data import DataLoader, TensorDataset

from utils.corset import Coreset


def transform_flatten(x):
    return x.flatten()


def merge_coresets(*coresets):
    merged_coreset = Coreset(coresets[0].coreset_size, coresets[0].method)

    merged_coreset.tensors = tuple(
        map(torch.cat, zip(*map(lambda x: x.tensors, coresets)))
    )
    return merged_coreset


def weighted_kl_divergence(posterior: Normal, prior: Normal):
    kl = kl_divergence(posterior, prior)

    v = prior.variance.sum(dim=-1)
    weights = torch.softmax(-v, dim=0)

    return kl.sum(dim=-1) * weights


def to_tensor_dataset(dataset):
    loader = DataLoader(dataset, batch_size=len(dataset))
    images, labels = next(iter(loader))

    return TensorDataset(images, labels)
