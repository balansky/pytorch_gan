from torchvision.models import inception_v3
from utils.sample import sample_noises
import torch
import math


class Inception(object):

    def __init__(self, gen, n_images=50000, batch_size=100, splits=10, device=torch.device("cpu")):
        self.gen = gen
        self.n_images = n_images
        self.batch_size = batch_size
        self.splits = splits
        self.n_batches = int(math.ceil(float(n_images)/float(batch_size)))
        self.inception_model = inception_v3(pretrained=True, aux_logits=False)
        self.inception_model.eval().to(device)


    def forward(self):
        pass