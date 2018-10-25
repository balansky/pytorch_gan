import torch
import copy

def loss_hinge_dis(dis_fake, dis_real):
    loss = torch.nn.functional.relu(1.0 - dis_real).mean() + \
           torch.nn.functional.relu(1.0 + dis_fake).mean()
    return loss

def loss_hinge_gen(dis_fake):
    loss = -dis_fake.mean()
    return loss


class GanUpdater(object):

    def __init__(self, genenerator, discriminator, gen_optimizer, dis_optimizer, dataset, n_gen_samples,
                 n_dis, loss_type, decay_gamma, device):
        self.gen = genenerator.to(device)
        self.dis = discriminator.to(device)
        self.mirror_gen = copy.deepcopy(genenerator).to(device)
        self.mirror_gen.eval()
        self.gen_optimizer = gen_optimizer
        self.dis_optimizer = dis_optimizer
        self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(gen_optimizer, gamma=decay_gamma)
        self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(dis_optimizer, gamma=decay_gamma)
        self.num_categories = genenerator.n_categories
        self.dataset = dataset
        self.batch_size = dataset.batch_size
        self.z_dim = genenerator.z_dim
        self.device = device
        self.n_dis = n_dis
        self.n_gen_samples = n_gen_samples
        self.fixed_noise, self.fixed_y = self.sample_fakes(n_gen_samples)
        if loss_type == "hinge":
            self.loss_gen = loss_hinge_gen
            self.loss_dis = loss_hinge_dis
        else:
            raise NotImplementedError

    def sample_fakes(self, batch_size):
        noise = torch.randn(batch_size, self.z_dim, device=self.device)
        if self.num_categories:
            y_fake = torch.randint(low=0, high=self.num_categories, size=(batch_size,), dtype=torch.long,
                                   device=self.device)
        else:
            y_fake = None
        return noise, y_fake

    def gen_samples(self):
        with torch.no_grad():
            self.mirror_gen.load_state_dict(self.gen.state_dict())
            fake = (self.mirror_gen(self.fixed_noise, self.fixed_y).detach().cpu()) * .5 + .5
        return fake

    def save(self, gen_path, dis_path):
        torch.save(self.gen.state_dict(), gen_path)
        torch.save(self.dis.state_dict(), dis_path)

    def load(self, gen_path, dis_path):
        self.gen.load_state_dict(torch.load(gen_path))
        self.dis.load_state_dict(torch.load(dis_path))

    def update(self, iter):

        for _ in range(self.n_dis):
            x_real, y_real = self.dataset.get_next()
            x_real = x_real.to(self.device)
            y_real = y_real.to(self.device)
            noise, y_fake = self.sample_fakes(self.batch_size)
            self.dis_optimizer.zero_grad()
            self.gen_optimizer.zero_grad()
            dis_real = self.dis(x_real, y_real)
            dis_fake = self.dis(self.gen(noise, y_fake).detach(), y_fake)
            disc_loss = self.loss_dis(dis_fake, dis_real)
            disc_loss.backward()
            self.dis_optimizer.step()

        self.dis_optimizer.zero_grad()
        self.gen_optimizer.zero_grad()

        noise, y_fake = self.sample_fakes(self.batch_size)

        x_fake = self.gen(noise, y_fake)
        gen_loss = -self.dis(x_fake, y_fake).mean()
        gen_loss.backward()

        self.gen_optimizer.step()

        if iter % len(self.dataset) == 0:
            self.scheduler_g.step()
            self.scheduler_d.step()

        return disc_loss, gen_loss
