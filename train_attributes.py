import argparse
import glob

import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torchfunc
import torchvision.transforms as transforms
import numpy as np
import random
import os
import cv2

from models.model import Generator

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None

def get_args():
    parser = argparse.ArgumentParser(description="training script",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--batch_size", "-b", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")

    # Generator stuff
    parser.add_argument("--generator_ckpt", type=str, default=None, help="path to the generator checkpoint")
    parser.add_argument("--size", type=int, default=256, help="image sizes for the model")
    parser.add_argument("--latent", type=int, default=512, help="latent vector dim")
    parser.add_argument("--n_mlp", type=int, default=8, help="number of layers in Z to W mapping")

    return parser.parse_args()

class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device

        self.generator = Generator(
            args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
        ).to(device)

        self.attribute_classifier = None

        if args.generator_ckpt is not None:
            print("load model:", args.generator_ckpt)
            generator_checkpoint = torch.load(args.generator_ckpt) #, map_location=lambda storage, loc: storage)
            self.generator.load_state_dict(generator_checkpoint['g_ema'], strict=False) # TODO: Maybe need to load g



    def _do_epoch(self, epoch_idx):
        pass

    def do_test(self):
        pass

    def do_training(self):
        pass

def main():
    args = get_args()

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = Trainer(args, device)
    trainer.do_training()

if __name__ == "__main__":
    torchfunc.cuda.reset()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    main()