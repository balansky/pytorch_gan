import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Parameter


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


# def weight_bar(w, u, pi):
#     # _, sigma, _ = torch.svd(w)
#     # sigma = sigma[0]
#
#     w_mat = w.data.view(w.data.shape[0], -1)
#
#     for _ in range(pi):
#         v = l2normalize(torch.mv(torch.t(w_mat), u))
#
#         u = l2normalize(torch.mv(w_mat, v))
#
#     sigma = torch.dot(torch.mv(torch.t(w_mat), u), v)
#     w_bar = w / sigma
#
#     return w_bar, u, sigma


class SpectralNorm(torch.nn.Module):

    def __init__(self, out_features, power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.power_iterations=power_iterations
        self.out_features = out_features
        self.register_buffer("u", torch.randn(out_features, requires_grad=False))

    def forward(self, input):
        # _, sigma, _ = torch.svd(input)
        # sigma = sigma[0]

        w = input

        w_mat = w.view(w.data.shape[0], -1)

        with torch.no_grad():
            for _ in range(self.power_iterations):
                v = l2normalize(torch.mv(torch.t(w_mat), self.u))

                self.u = l2normalize(torch.mv(w_mat, v))

        sigma = self.u.dot(w_mat.mv(v))

        w_bar = w / sigma

        return w_bar, sigma


class Linear(torch.nn.Linear):

    def __init__(self, *args, spectral_norm_pi=1, **kwargs):
        super(Linear, self).__init__(*args, **kwargs)
        if spectral_norm_pi > 0:
            self.add_module("sn", SpectralNorm(self.weight.shape[0], spectral_norm_pi))
        else:
            self.add_module("sn", None)


    def forward(self, input):
        if self.sn:
            w_bar, sigma = self.sn(self.weight)
            # self.w_bar = w_bar
            # self.sigma = sigma
        else:
            w_bar = self.weight
        return F.linear(input, w_bar, self.bias)


class Conv2d(torch.nn.Conv2d):

    def __init__(self, *args, spectral_norm_pi=1, **kwargs):
        super(Conv2d, self).__init__(*args, **kwargs)
        if spectral_norm_pi > 0:
            self.add_module("sn", SpectralNorm(self.weight.shape[0], spectral_norm_pi))
        else:
            self.add_module("sn", None)

    def forward(self, input):
        if self.sn:
            w_bar, sigma = self.sn(self.weight)
        else:
            w_bar = self.weight
        return F.conv2d(input, w_bar, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class Embedding(torch.nn.Embedding):

    def __init__(self, *args, spectral_norm_pi=1, **kwargs):
        super(Embedding, self).__init__(*args, **kwargs)
        if spectral_norm_pi > 0:
            self.add_module("sn", SpectralNorm(self.weight.shape[0], spectral_norm_pi))
        else:
            self.add_module("sn", None)

    def forward(self, input):
        if self.sn:
            w_bar, sigma = self.sn(self.weight)
        else:
            w_bar = self.weight
        return F.embedding(
            input, w_bar, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)


# class SpectralNorm(nn.Module):
#     def __init__(self, module, name='weight', power_iterations=1):
#         super(SpectralNorm, self).__init__()
#         self.module = module
#         self.name = name
#         self.power_iterations = power_iterations
#         self.sigma = None
#         self.w_bar = None
#         self.register_buffer("u", torch.randn(getattr(module, name).data.shape[0], requires_grad=False))
#
#
#     def _update_u_v(self):
#
#         w = getattr(self.module, self.name)
#         _, sigma, _ = torch.svd(w.data)
#         self.sigma = sigma[0]
#
#         self.w_bar = w / self.sigma.expand_as(w)
#
#         w.data = self.w_bar
#
#         # w_mat = w.data.view(w.data.shape[0], -1)
#
#         # for _ in range(self.power_iterations):
#         #     v = l2normalize(torch.mv(torch.t(w_mat), self.u))
#         #
#         #     self.u = l2normalize(torch.mv(w_mat, v))
#         #
#         # self.sigma = torch.dot(torch.mv(torch.t(w_mat), self.u), v)
#         # w.data = w.data/self.sigma
#
#         # setattr(self.module, self.name, w)
#
#
#     def forward(self, input):
#         # self._update_u_v()
#         w = getattr(self.module, self.name)
#         _, sigma, _ = torch.svd(w.data)
#         self.sigma = sigma[0]
#
#         self.w_bar = w / self.sigma.expand_as(w)
#         return torch.nn.functional.linear(input, self.w_bar, self.module.bias)
        # return self.module.forward(*args)