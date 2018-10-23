import numpy as np
from torchvision import datasets, transforms, utils
from nets.resnet import *


def train_cifar():
    device = torch.device("cuda:0")
    nz = 128
    lr = 0.0002
    num_epochs = 150

    iters = 0
    epoch = 0
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
                                              drop_last=True)

    netD = ResnetDiscriminator32(n_categories=10).to(device)
    netG = ResnetGenerator32(z_dim=nz, n_categories=10).to(device)

    evalG = ResnetGenerator32(z_dim=nz, n_categories=10).to(device)
    evalG.load_state_dict(netG.state_dict())
    evalG.eval()

    fixed_noise = torch.randn(25, nz, device=device)
    noise_y = torch.from_numpy(np.random.random_integers(low=0, high=9, size=25)).to(device)

    optimizerD = torch.optim.Adam(filter(lambda p: p.requires_grad, netD.parameters()), lr=lr, betas=(0., 0.9))

    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0., 0.9))

    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optimizerD, gamma=0.99)
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optimizerG, gamma=0.99)

    while True:
        epoch += 1

        for i, data in enumerate(data_loader):

            real_data = data[0].to(device)
            y = data[1].to(device)

            for _ in range(disc_iters):
                noise = torch.randn(batch_size, nz, device=device)
                optimizerD.zero_grad()
                output = netD(real_data, y)
                disc_loss = torch.nn.functional.relu(1.0 - output).mean() + \
                            torch.nn.functional.relu(1.0 + netD(netG(noise, y), y)).mean()
                disc_loss.backward()
                optimizerD.step()

            optimizerD.zero_grad()
            optimizerG.zero_grad()

            noise = torch.randn(batch_size, nz, device=device)
            fake = netG(noise, y)
            gen_loss = -netD(fake).mean()
            gen_loss.backward()

            optimizerG.step()

            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                      % (epoch, num_epochs, i, len(data_loader),
                         disc_loss.item(), gen_loss.item()))

            iters += 1
            if iters >= max_iters:
                break
        scheduler_d.step()
        scheduler_g.step()
        with torch.no_grad():
            evalG.load_state_dict(netG.state_dict())
            # evalG.eval()
            fake = (evalG(fixed_noise, noise_y).detach().cpu()) * .5 + .5
            utils.save_image(fake, "/home/andy/Pictures/GAN/cifar/%d_img.png" % (epoch + 1), nrow=5, padding=2)

        if iters >= max_iters:
            break


train_cifar()