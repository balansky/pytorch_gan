import torch

class LinearDecayLR(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, decay_start, max_iterations, last_epoch=-1):
        self.decay_start = decay_start
        self.max_iterations = max_iterations
        super(LinearDecayLR, self).__init__(optimizer, last_epoch)
        self.step_gamma = [base_lr / (max_iterations - decay_start) for base_lr in self.base_lrs]

    def get_lr(self):
        if self.last_epoch < self.decay_start:
            return self.base_lrs
        else:
            return [(base_lr - step_gamma*(self.last_epoch - self.decay_start + 1)) for step_gamma, base_lr
                    in zip(self.step_gamma, self.base_lrs)]
