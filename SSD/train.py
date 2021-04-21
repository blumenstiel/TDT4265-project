import argparse
import logging
import torch
import pathlib
import numpy as np
from ssd.engine.inference import do_evaluation
from ssd.config.defaults import cfg
from ssd.data.build import make_data_loader
from ssd.engine.trainer import do_train
from ssd.modeling.detector import SSDDetector
from ssd.utils.checkpoint import CheckPointer
from ssd.utils.logger import setup_logger
from ssd import torch_utils


from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)
np.random.seed(0)
torch.manual_seed(0)

# Allow torch/cudnn to optimize/analyze the input/output shape of convolutions
# To optimize forward/backward pass.
# This will increase model throughput for fixed input shape to the network
torch.backends.cudnn.benchmark = True

# Cudnn is not deterministic by default. Set this to True if you want
# to be sure to reproduce your results
torch.backends.cudnn.deterministic = True


def start_train(cfg):
    logger = logging.getLogger('SSD.trainer')
    model = SSDDetector(cfg)
    model = torch_utils.to_cuda(model)

    if cfg.SOLVER.OPT == 'SGD':
        print('Optimizer: SGD')
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.SOLVER.LR,
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY
        )
    elif cfg.SOLVER.OPT == 'Adam':
        print('Optimizer: Adam')
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.SOLVER.LR
        )

    # Add scheduler for adjusting learning rate during training
    if cfg.SOLVER.LR_SCHEDULER == 'LambdaLR':
        # set lambda function
        lambda_func = lambda epoch: cfg.SOLVER.LAMBDA ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_func)
    if cfg.SOLVER.LR_SCHEDULER == 'OneCycleLR':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.SOLVER.LR * 10,
                                                        steps_per_epoch=1, epochs=cfg.SOLVER.MAX_ITER)
    if cfg.SOLVER.LR_SCHEDULER == 'CyclicLR':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=cfg.SOLVER.LR_MIN, max_lr=cfg.SOLVER.LR,
                                                      step_size_up=cfg.SOLVER.STEP_SIZE, mode="triangular2")
    if cfg.SOLVER.LR_SCHEDULER == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=int(cfg.SOLVER.MAX_ITER/1000), eta_min=0)

    arguments = {"iteration": 0}
    save_to_disk = True
    checkpointer = CheckPointer(
        model, optimizer, cfg.OUTPUT_DIR, save_to_disk, logger,
        )
    extra_checkpoint_data = checkpointer.load()
    arguments.update(extra_checkpoint_data)

    max_iter = cfg.SOLVER.MAX_ITER
    train_loader = make_data_loader(cfg, is_train=True, max_iter=max_iter, start_iter=arguments['iteration'])

    model = do_train(
        cfg, model, train_loader, optimizer, scheduler,
        checkpointer, arguments)
    return model


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
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def main():
    args = get_parser().parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    output_dir = pathlib.Path(cfg.OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True, parents=True)

    logger = setup_logger("SSD", output_dir)
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    model = start_train(cfg)

    logger.info('Start evaluating...')
    torch.cuda.empty_cache()  # speed up evaluating after training finished
    do_evaluation(cfg, model)


if __name__ == '__main__':
    main()
