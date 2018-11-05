from torch.utils.data.dataloader import DataLoader, _DataLoaderIter, default_collate


class Dataset(DataLoader):

    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, collate_fn=default_collate, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        super(Dataset, self).__init__(dataset, batch_size, shuffle, sampler, batch_sampler,
                                      num_workers, collate_fn, pin_memory, drop_last,
                                      timeout, worker_init_fn)
        self.epochs = 1
        self.data_iter = None

    def reinitialize_iter(self):
        self.data_iter = iter(_DataLoaderIter(self))

    def get_next(self):
        try:
            if not self.data_iter:
                self.data_iter = iter(_DataLoaderIter(self))
            data = next(self.data_iter)
        except StopIteration:
            self.reinitialize_iter()
            self.epochs += 1
            return self.get_next()
        return data

    def __iter__(self):
        return iter(_DataLoaderIter(self))