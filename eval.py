import yaml
import argparse
from utils import yaml_utils
from utils.load import *
from training.evaluator import Inception


def main(args):
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config = yaml_utils.Config(yaml.load(open(args.config_path)))
    gen, _ = load_gan_model(config)
    gen.load_state_dict(torch.load(args.model_path))
    gen.eval().to(device)
    evaluator = Inception(n_images=args.n_eval, batch_size=args.batch_size, splits=args.splits, device=device)
    print("Evaluating Inception Score....")
    kl_score, kl_std = evaluator.eval_gen(gen)
    print("Inception Score: %.4f, Std: %.4f" % (kl_score, kl_std))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./results/gans',
                        help='saved model path')
    parser.add_argument('--config_path', type=str, default='configs/sn_cifar10_conditional.yml',
                        help='model configuration file')
    parser.add_argument('--batch_size', type=int, default=100, help="evaluation batch size(default:100)")
    parser.add_argument('--splits', type=int, default=10, help="splits for inception score(default: 10)")
    parser.add_argument('--n_eval', type=int, default=50000, help="total number of evaluations(default:50000)")
    parser.add_argument('--device', type=str, default=None, help="cpu or gpu")
    args = parser.parse_args()
    main(args)
