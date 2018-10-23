import torch
from torch import nn
from torch.nn import Parameter


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        self.w = getattr(module, name)
        self.sigma = None
        self.output_dim = self.w.data.shape[0]
        self.u = Parameter(self.w.data.new(self.output_dim).normal_(0, 1), requires_grad=False)


    def _update_u_v(self):
        w_reshaped = self.w.data.view(self.output_dim, -1)

        for _ in range(self.power_iterations):
            v = l2normalize(torch.mv(torch.t(w_reshaped), self.u.data))

            self.u.data = l2normalize(torch.mv(w_reshaped, v))

        self.sigma = torch.dot(torch.mv(torch.t(w_reshaped), self.u.data), v)
        self.w.data = self.w.data/self.sigma.expand_as(self.w.data)


    def forward(self, *args):
        self._update_u_v()
        return self.module(*args)