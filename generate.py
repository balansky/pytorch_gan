import yaml
import argparse
from utils import yaml_utils
from utils.load import *
from utils.sample import sample_noises
import torchvision
import numpy as np
import matplotlib.pyplot as plt


def main(args):
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config = yaml_utils.Config(yaml.load(open(args.config_path)))
    gen, _ = load_gan_model(config)
    gen.load_state_dict(torch.load(args.model_path))
    gen.eval().to(device)
    if not args.g_category:
        batch_noise, batch_labels = sample_noises(args.n_samples, gen.z_dim, gen.n_categories, device)
    else:
        batch_noise, _ = sample_noises(args.n_samples, gen.z_dim, device=device)
    batch_labels = batch_noise.new_full((args.n_samples,), fill_value=args.g_category, dtype=torch.long)
    samples = gen(batch_noise, batch_labels).detach().cpu() * .5 + .5
    grid = torchvision.utils.make_grid(samples).numpy()
    grid = np.transpose(grid, (1, 2, 0))
    plt.imshow(grid)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./results/gans',
                        help='saved model path')
    parser.add_argument('--config_path', type=str, default='configs/sn_cifar10_conditional.yml',
                        help='model configuration file')
    parser.add_argument('--g_category', type=int, default=None, help="category index to generate")
    parser.add_argument('--n_samples', type=int, default=64, help="number of samples to generate")
    parser.add_argument('--device', type=str, default=None, help="cpu or gpu")

    args = parser.parse_args()
    main(args)