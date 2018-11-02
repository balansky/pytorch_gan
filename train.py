import yaml
import argparse
from utils import yaml_utils
from utils.load import *
from training.trainer import GanTrainer


def main(args):
    device = torch.device("cuda:0")
    config = yaml_utils.Config(yaml.load(open(args.config_path)))
    gen, dis = load_gan_model(config)
    gen_optimizer = load_optimizer(config, gen.parameters())
    dis_optimizer = load_optimizer(config, filter(lambda p: p.requires_grad, dis.parameters()))

    scheduler_g = load_scheduler(config, gen_optimizer)
    scheduler_d = load_scheduler(config, dis_optimizer)


    dataset = load_dataset(args.batch_size, args.data_dir, args.loaderjob, config)

    evaluator = load_evaluator(config, device)

    trainer = GanTrainer(args.iterations, dataset, gen, dis, gen_optimizer, dis_optimizer, args.result_dir,
                         scheduler_g, scheduler_d, evaluator=evaluator, device=device, **config.trainer['args'])

    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/base.yml', help='path to config file')
    parser.add_argument('--data_dir', type=str, default='./data/imagenet')
    parser.add_argument('--iterations', type=int, default=250000)
    parser.add_argument('--result_dir', type=str, default='./results/gans',
                        help='directory to save the results to')
    parser.add_argument('--batch_size', type=int, default=64, help="mini batch size")
    parser.add_argument('--loaderjob', type=int, default=4,
                        help='number of parallel data loading processes')
    args = parser.parse_args()
    main(args)