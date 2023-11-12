from torch.optim.lr_scheduler import LinearLR as BaseLinearLR


class LinearLR(BaseLinearLR):
    def __init__(self, optimizer, start_factor=1.0 / 3, end_factor=1.0,
                 total_iters=5, last_epoch=-1, start_epoch=0, verbose=False):
        self.start_epoch = start_epoch
        super().__init__(optimizer, start_factor, end_factor,
                         total_iters, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch <= self.start_epoch:
            return [group['lr'] * self.start_factor for group in self.optimizer.param_groups]

        if self.last_epoch > self.total_iters + self.start_epoch:
            return [group['lr'] for group in self.optimizer.param_groups]

        return [group['lr'] * (1. + (self.end_factor - self.start_factor) /
                (self.total_iters * self.start_factor + (self.last_epoch - self.start_epoch - 1) *
                 (self.end_factor - self.start_factor)))
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [base_lr * (self.start_factor +
                           (self.end_factor - self.start_factor) *
                           min(self.total_iters,
                               max(self.last_epoch - self.start_epoch, 0)) / self.total_iters)
                for base_lr in self.base_lrs]
