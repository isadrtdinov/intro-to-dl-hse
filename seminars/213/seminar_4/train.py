import argparse

import torch
import wandb
from torch.utils.data import DataLoader

import sys
sys.path.append(".")

from src import config
from src.loops import train
from src.utils import set_random_seed, checkpoint


parser = argparse.ArgumentParser(description="Train pix2pix GAN model.")
parser.add_argument("-c", "--config", metavar="CONFIG", type=str, default="sat2map",
                    help="Config filename (default: %(default)s).")
parser.add_argument("-pg", "--gen_pretrained", type=str, default="none",
                    help="Pretrained generator weights (default: %(default)s).")
parser.add_argument("-pd", "--dis_pretrained", type=str, default="none",
                    help="Pretrained discriminator weights (default: %(default)s).")
parser.add_argument("-r", "--resume", action="store_true",
                    help="Resume training from checkpoint.")
parser.add_argument("-s", "--seed", metavar="INT", type=int, default=42,
                    help="Random seed (default: %(default)s).")
parser.add_argument("--n_tries", metavar="INT", type=int, default=10,
                    help="Number of times to restart training in case of failure (default: %(default)s).")
parser.add_argument("--progress", choices=["epochs", "samples"], default="epochs",
                    help="How to draw progressbars (default: %(default)s).")
args = parser.parse_args()

config_ = getattr(config, args.config)

data_config = config_.DataConfig()
model_config = config_.ModelConfig()
train_config = config_.TrainConfig()

resume = args.resume


set_random_seed(args.seed)

train_loader = DataLoader(
    data_config.train_dataset,
    batch_size=train_config.train_batch,
    shuffle=True,
    num_workers=train_config.num_workers
)

valid_loader = DataLoader(
    data_config.valid_dataset,
    batch_size=train_config.valid_batch,
    shuffle=False,
    num_workers=train_config.num_workers
)


generator = model_config.generator
generator = generator.to(train_config.device)

if args.gen_pretrained != "none":
    checkpoint.load_pretrained(args.gen_pretrained, "generator", generator)

discriminator = model_config.discriminator
discriminator = discriminator.to(train_config.device)

if args.dis_pretrained != "none":
    checkpoint.load_pretrained(args.dis_pretrained, "discriminator", discriminator)

gen_optimizer = torch.optim.Adam(
    generator.parameters(),
    lr=train_config.gen_lr,
    betas=train_config.betas
)
dis_optimizer = torch.optim.Adam(
    discriminator.parameters(),
    lr=train_config.dis_lr,
    betas=train_config.betas
)

gen_scheduler = train_config.gen_scheduler(gen_optimizer)
dis_scheduler = train_config.dis_scheduler(dis_optimizer)

if resume:
    try:
        id = checkpoint.get_id(train_config.run_name)
    except FileNotFoundError:
        id = train_config.run_name
else:
    id = wandb.util.generate_id()
    checkpoint.store_id(train_config.run_name, id)
wandb.init(config=train_config, project=train_config.wandb_project,
           id=id, name=train_config.run_name, resume="allow")

for _ in range(args.n_tries):
    reply = train(train_config, train_loader, valid_loader,
                  generator, discriminator, gen_optimizer, dis_optimizer,
                  gen_scheduler, dis_scheduler, resume, args.progress)
    if reply == "success":
        exit()
    elif reply == "gradient explosion":
        print("Gradient exploded, restarting from latest checkpoint")
        resume = True

print("Maximum number of tries exceeded, aborting run")
