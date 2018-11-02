import os
import math
from torchvision import utils
import torch
import copy
from utils.sample import sample_noises
from utils.losses import loss_hinge_dis, loss_hinge_gen


class GanTrainer(object):

    def __init__(self, iteration, dataset, genenerator, discriminator, gen_optimizer, dis_optimizer, output_dir,
                 scheduler_g=None, scheduler_d=None, evaluator=None, n_gen_samples=64, n_dis=5, loss_type='hinge',
                 display_interval=100, snapshot_interval=1000, evaluation_interval=1000, device=torch.device('cpu')):
        self.gen = genenerator.to(device)
        self.dis = discriminator.to(device)
        self.mirror_gen = copy.deepcopy(genenerator).to(device)
        self.mirror_gen.eval()
        self.gen_optimizer = gen_optimizer
        self.dis_optimizer = dis_optimizer
        self.scheduler_g = scheduler_g
        self.scheduler_d = scheduler_d
        self.num_categories = genenerator.n_categories
        self.dataset = dataset
        self.batch_size = dataset.batch_size
        self.z_dim = genenerator.z_dim
        self.device = device
        self.n_dis = n_dis
        self.n_gen_samples = n_gen_samples
        self.fixed_noise, self.fixed_y = sample_noises(self.n_gen_samples, self.z_dim, self.num_categories, device)
        if loss_type == "hinge":
            self.loss_gen = loss_hinge_gen
            self.loss_dis = loss_hinge_dis
        else:
            raise NotImplementedError

        self.iteration = iteration
        self.evaluator = evaluator
        self.display_interval = display_interval
        self.snapshot_interval = snapshot_interval
        self.evaluation_interval = evaluation_interval
        self.output_dir = output_dir
        self.snapshot_dir = os.path.join(output_dir, 'snapshots')
        self.sample_dir = os.path.join(output_dir, 'samples')
        self.n_row = max(int(math.sqrt(n_gen_samples)), 1)
        self.fixed_noise, self.fixed_y = sample_noises(n_gen_samples, self.z_dim, self.num_categories, device)

    def create_snapshot_dir(self):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        if not os.path.exists(self.snapshot_dir):
            os.mkdir(self.snapshot_dir)
        if not os.path.exists(self.sample_dir):
            os.mkdir(self.sample_dir)

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

    def update(self):

        for _ in range(self.n_dis):
            x_real, y_real = self.dataset.get_next()
            x_real = x_real.to(self.device)
            y_real = y_real.to(self.device)
            noise, y_fake = sample_noises(self.batch_size, self.z_dim, self.num_categories, self.device)
            self.dis_optimizer.zero_grad()
            self.gen_optimizer.zero_grad()
            dis_real = self.dis(x_real, y_real)
            dis_fake = self.dis(self.gen(noise, y_fake).detach(), y_fake)
            disc_loss = self.loss_dis(dis_fake, dis_real)
            disc_loss.backward()
            self.dis_optimizer.step()

        self.dis_optimizer.zero_grad()
        self.gen_optimizer.zero_grad()

        noise, y_fake = sample_noises(self.batch_size, self.z_dim, self.num_categories, self.device)

        x_fake = self.gen(noise, y_fake)
        gen_loss = -self.dis(x_fake, y_fake).mean()
        gen_loss.backward()

        self.gen_optimizer.step()

        if self.scheduler_d and self.scheduler_g:
            self.scheduler_g.step()
            self.scheduler_d.step()

        return disc_loss, gen_loss

    def run(self):
        self.create_snapshot_dir()
        for i in range(1, self.iteration + 1):
            disc_loss, gen_loss = self.update()
            if i % self.display_interval == 0 or i == self.iteration:
                print('[%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                      % (i, disc_loss.item(), gen_loss.item()))
            if i % self.snapshot_interval == 0 or i == self.iteration:

                fake = self.gen_samples()
                utils.save_image(fake, os.path.join(self.sample_dir, "%d_img.png" % i), nrow=self.n_row, padding=2)
                self.save(os.path.join(self.snapshot_dir, "gen_%d.pt" % i),
                          os.path.join(self.snapshot_dir, "dis_%d.pt" % i))
            if self.evaluator and (i % self.evaluation_interval == 0 or i == self.iteration):
                self.evaluator.eval()
        print("Training Done !")


