
"""
Use:

!python transfer_model.py <config/path/rdd_model> <model_name> <new_dir>

model_name: specify model to copy

new_dir: name directory in drive

this script is creating a config file

e.g.
!python transfer_model.py configs/train_rdd2020_colab_resnext.yaml model_010000.pth transfer_300_Triangular

!python train.py /content/drive/MyDrive/transfer_300_Triangular/config_transfer_300_Triangular.yaml

"""

import pathlib
import torch
import os
from ssd.config.defaults import cfg
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With PyTorch')
    parser.add_argument(
        "config_file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "model_name",
        default="model_final",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "new_dir",
        default="",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    output_dir = pathlib.Path(cfg.OUTPUT_DIR)
    # Define new output directory to
    if args.new_dir == "":
        new_output_dir = pathlib.Path(
            output_dir.parent,
            output_dir.stem + "_tdt4265"
        )
    else:
        new_output_dir = pathlib.Path(
            '/content/drive/MyDrive/',
            args.new_dir
        )

    # Copy checkpoint
    if not os.path.isdir(new_output_dir):
        new_output_dir.mkdir()
    new_checkpoint_path = new_output_dir.joinpath("rdd2020_model.pth")
    previous_checkpoint_path = pathlib.Path(cfg.OUTPUT_DIR, args.model_name)
    assert previous_checkpoint_path.is_file(), f'Path is not a file: {previous_checkpoint_path}'
    # Only keep the parameters for the model
    new_checkpoint = {
        "model": torch.load(previous_checkpoint_path)["model"]
    }
    torch.save(new_checkpoint, str(new_checkpoint_path))
    del new_checkpoint

    with open(new_output_dir.joinpath("last_checkpoint.txt"), "w") as fp:
        fp.writelines(str(new_checkpoint_path))

    with open(args.config_file, "r") as fp:
        old_config_lines = fp.readlines()
    new_config_lines = []
    # Overwrite config values
    for line in old_config_lines:
        if line == '    TRAIN: ("rdd2020_train",)\n':
            old = line
            line = '    TRAIN: ("tdt4265_train",)\n'
            print(f"overwriting: {old} with {line}")
        if line == '    TRAIN: ("rdd2020_train_oversampling",)\n':
            old = line
            line = '    TRAIN: ("tdt4265_train",)\n'  # TODO Add oversamling for tdt dataset
            print(f"overwriting: {old} with {line}")
        if line == '    TEST: ("rdd2020_val", )\n':
            old = line
            line = '    TEST: ("tdt4265_val", )\n'
            print(f"overwriting: {old} with {line}")
        if line.startswith('OUTPUT_DIR:'):
            line = f'OUTPUT_DIR: {new_output_dir}\n'
        new_config_lines.append(line)
    new_config_path = new_output_dir.joinpath("config_" + new_output_dir.stem + ".yaml")
    with open(new_config_path, "w") as fp:
        fp.writelines(new_config_lines)
    print("New config saved to:", new_config_path)
    # print("Starting train")
    # os.system(f"python train.py {new_config_path}")
