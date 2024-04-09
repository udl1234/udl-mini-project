from torch.utils.data import TensorDataset
import torch


class Coreset(TensorDataset):

    def __init__(self, size, method):
        super().__init__(
            torch.tensor([], dtype=torch.float32), torch.tensor([], dtype=torch.long)
        )

        self.coreset_size = size
        self.method = method

        if method not in ["random", "k_center"]:
            raise ValueError(f"Unknown method: {method}")

    def with_method(self, train_set: TensorDataset):
        if self.method == "random":
            self.add_random(train_set)
        elif self.method == "k_center":
            self.add_k_center(train_set)

    def add_random(self, train_set: TensorDataset):
        N = len(train_set)
        assert N >= self.coreset_size, "Coreset size is larger than the dataset size"

        idxs = torch.randperm(N)[: self.coreset_size]
        self._update_coreset(train_set, idxs)

    def add_k_center(self, train_set: TensorDataset):
        N = len(train_set)
        assert N >= self.coreset_size, "Coreset size is larger than the dataset size"

        dists = torch.full((N,), float("inf"))
        # current = torch.randint(0, N, ())
        current = 0
        idxs = torch.empty(self.coreset_size, dtype=torch.long)

        for i in range(self.coreset_size):
            idxs[i] = current
            dists = self._update_distance(dists, train_set.tensors[0], current)
            current = torch.argmax(dists)

        self._update_coreset(train_set, idxs)

    def _update_distance(self, dists, data, current):
        current_data = data[current].unsqueeze(0)
        new_dists = torch.norm(data - current_data, p=2, dim=1)

        dists = torch.minimum(new_dists, dists)
        return dists

    def _update_coreset(self, train_set: TensorDataset, idxs: torch.Tensor):
        mask = torch.ones(len(train_set), dtype=torch.bool)
        mask[idxs] = False

        def move_data(data):
            coreset_tensor, train_tensor = data
            coreset_tensor = torch.cat([coreset_tensor, train_tensor[idxs]])
            train_tensor = train_tensor[mask]

            return coreset_tensor, train_tensor

        coreset_tensors, train_tensors = zip(
            *map(move_data, zip(self.tensors, train_set.tensors))
        )
        self.tensors = coreset_tensors
        train_set.tensors = train_tensors
