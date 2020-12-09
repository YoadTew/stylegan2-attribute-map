import argparse
import glob

import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torchfunc
import torchvision.transforms as transforms
from torchvision import utils
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import os
import cv2

from models.model import Generator
from models.resnet import resnet50


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def get_args():
    parser = argparse.ArgumentParser(description="training script",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--batch_size", "-b", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--log_dir", default='logs', help="Logs dir path")
    parser.add_argument("--n_sample", type=int, default=8, help="number of the samples generated during training")
    parser.add_argument("--iters", type=int, default=100, help="number of the iterations for every epoch")

    # Generator
    parser.add_argument("--generator_ckpt", type=str, default=None, help="path to the generator checkpoint")
    parser.add_argument("--size", type=int, default=256, help="image sizes for the model")
    parser.add_argument("--latent", type=int, default=512, help="latent vector dim")
    parser.add_argument("--n_mlp", type=int, default=8, help="number of layers in Z to W mapping")
    parser.add_argument("--channel_multiplier", type=int, default=2,
                        help="channel multiplier of the generator. config-f = 2, else = 1")

    parser.add_argument("--mixing", type=float, default=0., help="probability of latent code mixing")
    parser.add_argument("--truncation", type=float, default=0.7, help="truncation ratio")
    parser.add_argument("--truncation_mean", type=int, default=4096,
                        help="number of vectors to calculate mean for the truncation")

    # Attribute classifier
    parser.add_argument("--attribute_classifier_ckpt", type=str, default=None, help="path to the classifier checkpoint")

    return parser.parse_args()

class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device

        # Load generator
        self.generator = Generator(
            args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
        ).to(device)

        if args.generator_ckpt is not None:
            print("load generator:", args.generator_ckpt)
            generator_checkpoint = torch.load(args.generator_ckpt)  # map_location=lambda storage, loc: storage)
            self.generator.load_state_dict(generator_checkpoint['g_ema'], strict=False)  # TODO: Maybe need to load g

        # Load classifier
        resnet = resnet50(pretrained=True, num_classes=1)
        self.attribute_classifier = resnet.to(device)

        if args.attribute_classifier_ckpt is not None:
            print("load attribute clasifier:", args.attribute_classifier_ckpt)
            classifier_checkpoint = torch.load(args.attribute_classifier_ckpt)
            self.attribute_classifier.load_state_dict(classifier_checkpoint['state_dict'])

        self.optimizer = torch.optim.AdamW(self.generator.attribute_mapper.parameters(), lr=args.lr)

        self.classifier_criterion = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

        self.sample_z = None # Sample for validation
        self.mean_latent = None # Mean for truncation

        cudnn.benchmark = True
        self.writer = SummaryWriter(log_dir=str(args.log_dir))

    def _do_epoch(self, epoch_idx):
        self.generator.train()
        self.attribute_classifier.train()
        requires_grad(self.generator, True)
        requires_grad(self.attribute_classifier, True)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        for idx in range(self.args.iters):
            self.optimizer.zero_grad()
            self.generator.zero_grad()
            self.attribute_classifier.zero_grad()

            z_batch = mixing_noise(self.args.batch_size, self.args.latent, self.args.mixing, self.device)

            fake_img_batch, W, W_tag = self.generator(z_batch, truncation=self.args.truncation, truncation_latent=self.mean_latent,
                                         use_attribute_map=True, return_latents=True)

            fake_img_batch_without_attribute, _ = self.generator(z_batch, truncation=self.args.truncation, truncation_latent=self.mean_latent,
                                         use_attribute_map=False)

            fake_img_batch_norm = torch.zeros(fake_img_batch.shape, device=self.device)
            for i, fake_img in enumerate(fake_img_batch):
                fake_img_batch_norm[i] = normalize(fake_img)

            probs = self.attribute_classifier(fake_img_batch_norm)

            # Losses
            classifier_loss = self.classifier_criterion(probs, torch.ones(probs.shape, device=self.device))
            img_mse_loss = self.mse_loss(fake_img_batch, fake_img_batch_without_attribute)
            latent_mse_loss = self.mse_loss(W, W_tag)

            loss = 1 * img_mse_loss + (10**2) * latent_mse_loss + 1 * classifier_loss

            print(f'Loss iter {((epoch_idx-1) * self.args.iters) + idx}:  Classifier {classifier_loss}, MSE {latent_mse_loss}')

            loss.backward()
            self.optimizer.step()

        self.do_test(epoch_idx)

    def do_test(self, epoch_idx, use_attribute_map=True):
        self.generator.eval()
        self.attribute_classifier.eval()
        requires_grad(self.generator, False)
        requires_grad(self.attribute_classifier, False)

        with torch.no_grad():
            sample_img_batch, _ = self.generator([self.sample_z], truncation=self.args.truncation,
                                               truncation_latent=self.mean_latent,
                                               use_attribute_map=use_attribute_map)

            sample_img_batch = sample_img_batch.to('cpu')

            out_dir = f'experiment/{epoch_idx}/'

            if not os.path.exists(os.path.dirname(out_dir)):
                os.makedirs(os.path.dirname(out_dir))

            if use_attribute_map:
                for i, (original_img, sample_img) in enumerate(zip(self.original_sample, sample_img_batch)):
                    utils.save_image(
                        torch.stack([original_img, sample_img]),
                        f"{out_dir}/{str(i).zfill(6)}.png",
                        nrow=2,
                        normalize=True,
                        range=(-1, 1),
                    )
            else:
                for i, sample_img in enumerate(sample_img_batch):
                    utils.save_image(
                        sample_img,
                        f"{out_dir}/{str(i).zfill(6)}.png",
                        nrow=1,
                        normalize=True,
                        range=(-1, 1),
                    )

        return sample_img_batch


    def do_training(self):
        self.sample_z = torch.randn(self.args.n_sample, self.args.latent, device=self.device)

        if self.args.truncation < 1:
            with torch.no_grad():
                self.mean_latent = self.generator.mean_latent(self.args.truncation_mean)

        self.original_sample = self.do_test(0, use_attribute_map=False)

        for self.current_epoch in range(1, 10):
            self._do_epoch(self.current_epoch)

        self.writer.close()


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