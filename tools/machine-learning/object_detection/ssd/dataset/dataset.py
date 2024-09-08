from abc import ABC

from torch.utils.data import Dataset as TorchDataset


class DetectionDataset[Inner, Batch](TorchDataset, ABC):
    def __getitem__(self, index: int) -> Inner:
        return super().__getitem__(index)

    def n_classes(self) -> int:
        raise NotImplementedError

    def classes(self) -> list[str]:
        raise NotImplementedError

    def collate_fn(self, batch: list[Inner]) -> Batch:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
