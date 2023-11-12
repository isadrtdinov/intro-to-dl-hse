import argparse
import os

from functools import partial
from torch.utils.data import DataLoader

import sys
sys.path.append(".")

from src import config
from src.loops import predict
from src.utils import checkpoint, smooth_colors, save_image, split_extension


parser = argparse.ArgumentParser(description="Process images using trained pix2pix GAN model.")
parser.add_argument("-c", "--config", metavar="CONFIG", type=str, default="sat2map",
                    help="Config filename (default: %(default)s).")
parser.add_argument("-d", "--data_dir", type=str, default="auto",
                    help="Path to directory with input images (default: %(default)s).")
parser.add_argument("-s", "--save_dir", type=str, default="resources/predicted",
                    help="Where to save images (default: %(default)s).")
parser.add_argument("-pg", "--gen_pretrained", type=str, default="auto",
                    help="Pretrained generator weights (default: %(default)s).")
parser.add_argument("-e", "--enhance", action="store_true",
                    help="Apply post-processing for smooth color gradients.")
parser.add_argument("--kernel_size", type=int, default=5, metavar="INT",
                    help="Post-processing kernel size (default: %(default)s).")
parser.add_argument("--epsilon", type=float, default=0.99, metavar="FLOAT",
                    help="Post-processing epsilon (default: %(default)s).")
parser.add_argument("--lr", type=float, default=1e-3, metavar="FLOAT",
                    help="Post-processing step size (default: %(default)s).")
parser.add_argument("--num_steps", type=int, default=100, metavar="INT",
                    help="Post-processing number of steps (default: %(default)s).")
parser.add_argument("--unnorm", type=str, choices=["yes", "no"], default="yes",
                    help="Unnormalize images [-1; 1] -> [0, 1] when saving (default: %(default)s).")
args = parser.parse_args()

config_ = getattr(config, args.config)

data_config = config_.DataConfig()
model_config = config_.ModelConfig()
train_config = config_.TrainConfig()


if args.data_dir != "auto":
    data_config.valid_images_dir = args.data_dir

dataset = data_config.valid_dataset
predict_loader = DataLoader(
    dataset,
    batch_size=train_config.valid_batch,
    shuffle=False
)


generator = model_config.generator
generator = generator.to(train_config.device)

pretrained = args.gen_pretrained if args.gen_pretrained != "auto" \
                                 else train_config.run_name
checkpoint.load_pretrained(pretrained, "generator", generator)

post_process = partial(smooth_colors,
                       kernel_size=args.kernel_size,
                       epsilon=args.epsilon,
                       lr=args.lr,
                       num_steps=args.num_steps) if args.enhance \
                                                 else None

predicted = predict(generator, predict_loader, train_config.device, post_process)
save_dir = f"{args.save_dir}/{pretrained}"


def save_images(save_dir: str, images):
    os.makedirs(save_dir, exist_ok=True)

    for name, image in zip(dataset.ids, images):
        path = f"{save_dir}/{split_extension(os.path.basename(name))[0]}.png"
        save_image(path, image, args.unnorm == "yes")


if args.enhance:
    predicted, post_processed = predicted
    save_images(save_dir + "_e", post_processed)

save_images(save_dir, predicted)
