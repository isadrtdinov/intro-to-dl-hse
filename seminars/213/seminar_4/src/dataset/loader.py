class EpochLoader:
    """Arbitrary epoch size data loader.

    Args:
        loader: torch dataloader object
        epoch_size (int): num samples in one epoch

    """

    def __init__(
            self,
            loader,
            epoch_size
    ):
        self.loader = loader
        self.epoch_size = epoch_size

        self.sample_idx = 0
        self.iter_loader = iter(loader)

    def __next__(self):
        if self.sample_idx == self.epoch_size:
            self.sample_idx = 0
            raise StopIteration

        self.sample_idx += 1

        try:
            return next(self.iter_loader)
        except StopIteration:
            self.iter_loader = iter(self.loader)
            return next(self.iter_loader)

    def __len__(self):
        return self.epoch_size

    def __iter__(self):
        return self

    @property
    def dataset(self):
        return self.loader.dataset

    @dataset.setter
    def dataset(self, value):
        self.loader.dataset = value
