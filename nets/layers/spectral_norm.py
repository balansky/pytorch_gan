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
        self.sigma = None
        self.register_buffer("u", torch.randn(getattr(module, name).data.shape[0], requires_grad=False))


    def _update_u_v(self):

        w = getattr(self.module, self.name)
        # _, self.sigma, _ = torch.svd(w.data)
        # self.sigma = self.sigma[0]
        # w.data = w.data/self.sigma

        w_mat = w.data.view(w.data.shape[0], -1)

        for _ in range(self.power_iterations):
            v = l2normalize(torch.mv(torch.t(w_mat), self.u))

            self.u = l2normalize(torch.mv(w_mat, v))

        self.sigma = torch.dot(torch.mv(torch.t(w_mat), self.u), v)
        w.data = w.data/self.sigma
        # setattr(self.module, self.name, w)


    def forward(self, *args):
        self._update_u_v()
        return self.module(*args)