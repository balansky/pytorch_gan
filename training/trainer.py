import os
import math
from torchvision import utils


class GanTrainer(object):

    def __init__(self, updater, output_dir,
                 iteration, display_interval, snapshot_interval, evaluation_interval):
        self.updater = updater
        self.iteration = iteration
        self.display_interval = display_interval
        self.snapshot_interval = snapshot_interval
        self.evaluation_interval = evaluation_interval
        self.output_dir = output_dir
        self.snapshot_dir = os.path.join(output_dir, 'snapshots')
        self.sample_dir = os.path.join(output_dir, 'samples')
        self.n_row = max(int(math.sqrt(updater.n_gen_samples)), 1)

    def create_snapshot_dir(self):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        if not os.path.exists(self.snapshot_dir):
            os.mkdir(self.snapshot_dir)
        if not os.path.exists(self.sample_dir):
            os.mkdir(self.sample_dir)

    def run(self):
        self.create_snapshot_dir()
        for i in range(1, self.iteration + 1):
            disc_loss, gen_loss = self.updater.update()
            if i % self.display_interval == 0 or i == self.iteration:
                print('[%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                      % (i, disc_loss.item(), gen_loss.item()))
            if i % self.snapshot_interval == 0 or i == self.iteration:
                fake = self.updater.gen_samples()
                utils.save_image(fake, os.path.join(self.sample_dir, "%d_img.png" % i), nrow=self.n_row, padding=2)
                self.updater.save(os.path.join(self.snapshot_dir, "gen_%d.pt" % i),
                                  os.path.join(self.snapshot_dir, "dis_%d.pt" % i))
        print("Training Done !")



