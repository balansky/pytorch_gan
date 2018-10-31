import torch


def sample_noises(n_samples, noise_dim, n_categories=None, device=torch.device("cpu")):
    noise = torch.randn(n_samples, noise_dim, device=device)
    if n_categories:
        y_fake = torch.randint(low=0, high=n_categories, size=(n_samples,), dtype=torch.long,
                               device=device)
    else:
        y_fake = None
    return noise, y_fake