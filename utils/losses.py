import torch

def loss_hinge_dis(dis_fake, dis_real):
    loss = torch.nn.functional.relu(1.0 - dis_real).mean() + \
           torch.nn.functional.relu(1.0 + dis_fake).mean()
    return loss

def loss_hinge_gen(dis_fake):
    loss = -dis_fake.mean()
    return loss
