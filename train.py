import numpy as np
from torchvision import datasets, transforms, utils
from nets.resnet import *


class Dataset(object):

    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.data_iterator = iter(self.data_loader)

    def get_next(self):
        try:
            x_real, y_real = next(self.data_iterator)
        except Exception:
            self.data_iterator = iter(self.data_loader)
            return self.get_next()
        return x_real, y_real




def train_cifar():
    device = torch.device("cuda:0")
    nz = 128
    lr = 0.0002

    iters = 0
    max_iters = 50000
    batch_size = 64
    disc_iters = 5

    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    ])

    cifar_dataset = datasets.CIFAR10(root='/home/andy/Data/images/cifar10', train=True, download=True,
                                     transform=image_transform)
    data_loader = torch.utils.data.DataLoader(cifar_dataset, batch_size=batch_size, shuffle=True, num_workers=8,
                                              drop_last=True, pin_memory=True)
    data_iterator = Dataset(data_loader)
    netD = ResnetDiscriminator32(n_categories=10).to(device)
    netG = ResnetGenerator32(z_dim=nz, n_categories=10).to(device)

    evalG = ResnetGenerator32(z_dim=nz, n_categories=10).to(device)
    evalG.load_state_dict(netG.state_dict())
    evalG.eval()

    fixed_noise = torch.randn(25, nz, device=device)
    noise_y = torch.randint(low=0, high=10, size=(25,), dtype=torch.long, device=device)

    optimizerD = torch.optim.Adam(filter(lambda p: p.requires_grad, netD.parameters()), lr=lr, betas=(0., 0.9))

    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0., 0.9))

    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optimizerD, gamma=0.99)
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optimizerG, gamma=0.99)

    while True:
        iters += 1

        for _ in range(disc_iters):
            x_real, y_real = data_iterator.get_next()
            x_real = x_real.to(device)
            y_real = y_real.to(device)
            noise = torch.randn(batch_size, nz, device=device)
            y_fake = torch.randint(low=0, high=10, size=(batch_size,), dtype=torch.long, device=device)

            optimizerD.zero_grad()
            optimizerG.zero_grad()
            output = netD(x_real, y_real)
            disc_loss = torch.nn.functional.relu(1.0 - output).mean() + \
                        torch.nn.functional.relu(1.0 + netD(netG(noise, y_fake).detach(), y_fake)).mean()
            disc_loss.backward()
            optimizerD.step()

        optimizerD.zero_grad()
        optimizerG.zero_grad()

        noise = torch.randn(batch_size, nz, device=device)
        y_fake = torch.randint(low=0, high=10, size=(batch_size,), dtype=torch.long, device=device)
        x_fake = netG(noise, y_fake)
        gen_loss = -netD(x_fake, y_fake).mean()
        gen_loss.backward()

        optimizerG.step()

        if iters % 50 == 0:
            print('[%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                  % (iters, disc_loss.item(), gen_loss.item()))

        if iters % 1000 == 0:

            scheduler_d.step()
            scheduler_g.step()
            with torch.no_grad():
                evalG.load_state_dict(netG.state_dict())
                # evalG.eval()
                fake = (evalG(fixed_noise, noise_y).detach().cpu()) * .5 + .5
                utils.save_image(fake, "/home/andy/Pictures/GAN/cifar/%d_img.png" % iters, nrow=5, padding=2)

        if iters >= max_iters:
            break



train_cifar()