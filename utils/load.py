import torch
import os
import sys

Optimizer = {
    "adam": torch.optim.Adam
}


def load_module(fn, name):
    mod_name = os.path.splitext(os.path.basename(fn))[0]
    mod_path = os.path.dirname(fn)
    sys.path.insert(0, mod_path)
    return getattr(__import__(mod_name), name)


def load_model(model_fn, model_name, args=None):
    model = load_module(model_fn, model_name)
    if args:
        return model(**args)
    return model()


def load_optimizer(config, params):
    optimizer_config = config.optimizer
    optimizer = Optimizer[optimizer_config['name']]
    return optimizer(params, lr=optimizer_config['alpha'],
                     betas=(optimizer_config['beta1'], optimizer_config['beta2']))


def load_dataset(batch_size, root_dir, num_workers, config):
    dataset = load_module(config.dataset['dataset_fn'],
                          config.dataset['dataset_name'])
    return dataset(root=root_dir, batch_size=batch_size, num_workers=num_workers, **config.dataset['args'])


def load_gan_model(config):
    gen_config = config.models['generator']
    dis_config = config.models['discriminator']
    gen = load_model(gen_config['fn'], gen_config['name'], gen_config['args'])
    dis = load_model(dis_config['fn'], dis_config['name'], dis_config['args'])
    return gen, dis


def load_updater_class(config):
    return load_module(config.updater['fn'], config.updater['name'])