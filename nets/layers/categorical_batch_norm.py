from torch.nn import BatchNorm2d
import torch



class CategoricalBatchNorm(torch.nn.Module):

    def __init__(self, num_features, num_categories, eps=1e-5, momentum=0.1, affine=False,
                 track_running_stats=True):
        super(CategoricalBatchNorm, self).__init__()
        self.batch_norm = BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.gamma_c = torch.nn.Embedding(num_categories, num_features)
        self.beta_c = torch.nn.Embedding(num_categories, num_features)
        torch.nn.init.constant_(self.batch_norm.running_var.data, 0)
        torch.nn.init.constant_(self.gamma_c.weight.data, 1)
        torch.nn.init.constant_(self.beta_c.weight.data, 0)

    def forward(self, input, y):
        ret = self.batch_norm(input)
        gamma = self.gamma_c(y)
        beta = self.beta_c(y)
        gamma_b = gamma.unsqueeze(2).unsqueeze(3).expand_as(ret)
        beta_b = beta.unsqueeze(2).unsqueeze(3).expand_as(ret)
        return gamma_b*ret + beta_b
