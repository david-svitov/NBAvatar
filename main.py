import os
import argparse
import sys
import datetime

from omegaconf import OmegaConf

from src.utils.util import instantiate_from_config


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="Paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    return parser

if __name__ == '__main__':
    parser = get_parser()

    opt, unknown = parser.parse_known_args(sys.argv)

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    exp_name = opt.base[0].split('/')[-1].split('.')[0]
    time = datetime.datetime.now()
    time_stamp = time.strftime(f"%Y_%m-%d_%H-%M")
    log_folder_name = os.path.join(config.log_dir, exp_name, time_stamp)
    os.makedirs(log_folder_name, exist_ok=True)

    dataset = instantiate_from_config(config.dataset_train)
    trainer = instantiate_from_config(config.trainer)
    trainer.to(trainer.device)
    trainer.set_log_folder(log_folder_name)
    trainer.fit(dataset)
