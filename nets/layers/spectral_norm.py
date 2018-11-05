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

#
# class SpectralNorm(torch.nn.Module):
#
#     def __init__(self, out_features, power_iterations=1):
#         super(SpectralNorm, self).__init__()
#         self.power_iterations = power_iterations
#         self.out_features = out_features
#         # self.register_buffer("u", torch.randn(out_features, requires_grad=False))
#
#         self.register_buffer("u", torch.randn((1, out_features), requires_grad=False))
#
#     def forward(self, w):
#         w_mat = w.view(w.data.shape[0], -1)
#
#         # with torch.no_grad():
#         #     _, sigma, _ = torch.svd(w_mat)
#         #     sigma = sigma[0]
#
#         #
#         u = self.u
#         with torch.no_grad():
#             for _ in range(self.power_iterations):
#                 v = l2normalize(torch.mm(u, w_mat.data))
#
#                 u = l2normalize(torch.mm(v, torch.t(w_mat.data)))
#
#                 # v = l2normalize(torch.mv(torch.t(w_mat), self.u))
#
#                 # u = l2normalize(torch.mv(w_mat, v))
#
#         # sigma = u.dot(w_mat.mv(v))
#         sigma = torch.sum(torch.mm(u, w_mat) * v)
#
#         if self.training:
#             self.u = u
#         w_bar = torch.div(w, sigma)
#         # w_bar = w / sigma.expand_as(w.data)
#
#         return w_bar, sigma


def max_singular_value(w_mat, u, power_iterations):

    for _ in range(power_iterations):
        v = l2normalize(torch.mm(u, w_mat.data))

        u = l2normalize(torch.mm(v, torch.t(w_mat.data)))

    sigma = torch.sum(torch.mm(u, w_mat) * v)

    return u, sigma, v



class Linear(torch.nn.Linear):

    def __init__(self, *args, spectral_norm_pi=1, **kwargs):
        super(Linear, self).__init__(*args, **kwargs)
        self.spectral_norm_pi = spectral_norm_pi
        if spectral_norm_pi > 0:
            self.register_buffer("u", torch.randn((1, self.out_features), requires_grad=False))
        else:
            self.register_buffer("u", None)
        if self.bias is not None:
            torch.nn.init.constant_(self.bias.data, 0)


    def forward(self, input):
        if self.spectral_norm_pi > 0:
            w_mat = self.weight.view(self.out_features, -1)
            u, sigma, _ = max_singular_value(w_mat, self.u, self.spectral_norm_pi)

            # w_bar = torch.div(w_mat, sigma)
            w_bar = torch.div(self.weight, sigma)
            if self.training:
                self.u = u
            # self.w_bar = w_bar.detach()
            # self.sigma = sigma.detach()
        else:
            w_bar = self.weight
        return F.linear(input, w_bar, self.bias)


class Conv2d(torch.nn.Conv2d):

    def __init__(self, *args, spectral_norm_pi=1, **kwargs):
        super(Conv2d, self).__init__(*args, **kwargs)
        self.spectral_norm_pi = spectral_norm_pi
        if spectral_norm_pi > 0:
            self.register_buffer("u", torch.randn((1, self.out_channels), requires_grad=False))
        else:
            self.register_buffer("u", None)
        if self.bias is not None:
            torch.nn.init.constant_(self.bias.data, 0)

    def forward(self, input):
        if self.spectral_norm_pi > 0:
            w_mat = self.weight.view(self.out_channels, -1)
            u, sigma, _ = max_singular_value(w_mat, self.u, self.spectral_norm_pi)
            w_bar = torch.div(self.weight, sigma)
            if self.training:
                self.u = u
        else:
            w_bar = self.weight

        return F.conv2d(input, w_bar, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class Embedding(torch.nn.Embedding):

    def __init__(self, *args, spectral_norm_pi=1, **kwargs):
        super(Embedding, self).__init__(*args, **kwargs)
        self.spectral_norm_pi = spectral_norm_pi
        if spectral_norm_pi > 0:
            self.register_buffer("u", torch.randn((1, self.num_embeddings), requires_grad=False))
        else:
            self.register_buffer("u", None)

    def forward(self, input):
        if self.spectral_norm_pi > 0:
            w_mat = self.weight.view(self.num_embeddings, -1)
            u, sigma, _ = max_singular_value(w_mat, self.u, self.spectral_norm_pi)
            w_bar = torch.div(self.weight, sigma)
            if self.training:
                self.u = u
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