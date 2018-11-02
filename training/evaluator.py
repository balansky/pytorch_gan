from torchvision.models import inception_v3
from utils.sample import sample_noises
import torch
import math


class Inception(object):

    def __init__(self, n_images=50000, batch_size=100, splits=10, device=torch.device("cpu")):
        self.n_images = n_images
        self.batch_size = batch_size
        self.n_batches = int(math.ceil(float(n_images) / float(batch_size)))
        self.splits = splits
        self.n_batches = int(math.ceil(float(n_images)/float(batch_size)))
        self.device = device
        self.inception_model = inception_v3(pretrained=True, aux_logits=False)
        self.inception_model.eval().to(device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device)

    def generate_images(self, gen):
        batch_noise, batch_y = sample_noises(self.batch_size, gen.z_dim, gen.n_categories, self.device)
        batch_images = (gen(batch_noise, batch_y).detach() * .5 + .5)
        batch_images = torch.nn.functional.interpolate(batch_images, size=(299, 299), mode='bilinear')
        batch_images = (batch_images - self.mean) / self.std
        return batch_images

    def eval(self, gen):
        ys = []
        scores = []
        for i in range(self.n_batches):
            batch_images = self.generate_images(gen)
            y = self.inception_model(batch_images)
            ys.append(y)
        ys = torch.cat(ys, 0)
        for j in range(self.splits):
            part = ys[(j*self.n_images//self.splits): ((j+1)*self.n_images // self.splits), :]
            kl = part * (torch.log(part) - torch.log((torch.mean(part, 0))))
            kl = torch.mean(torch.sum(kl, 1))
            scores.append(kl)
        scores = torch.cat(scores, 0)
        m_scores = torch.mean(scores).detach().cpu().numpy()
        m_std = torch.std(scores).detach().cpu().numpy()
        return m_scores, m_std


