import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset

from dataclasses import dataclass
from torch.nn import L1Loss
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from ..dataset import OxfordIIITPetDataset
from ..models import RandomShift, UNet, PatchDiscriminator, init_weights
from ..loss import EdgeLoss, VGGPerceptualLoss, PreprocessWrapper
from ..metrics import NegativeLPIPS
from ..schedulers import LinearLR


@dataclass
class DataConfig:
    data_dir = "data/oxford-iiit-pet"
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.Normalize(2, 1)
    ])
    target_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256), antialias=True),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    @property
    def train_dataset(self) -> Dataset:
        return OxfordIIITPetDataset(self.data_dir,
                                    input_transform=self.input_transform,
                                    target_transform=self.target_transform,
                                    random_aug=True,
                                    test_mode=False)

    @property
    def valid_dataset(self) -> Dataset:
        return OxfordIIITPetDataset(self.data_dir,
                                    input_transform=self.input_transform,
                                    target_transform=self.target_transform)


@dataclass
class ModelConfig:
    gen_in_channels = 1
    gen_out_channels = 3
    gen_num_levels = 8
    gen_hidden_channels = 64
    gen_max_hidden_channels = 512

    dis_in_channels = 4
    dis_out_channels = 1
    dis_num_levels = 5
    dis_hidden_channels = 64

    @property
    def generator(self) -> nn.Module:
        model = UNet(self.gen_in_channels, self.gen_out_channels,
                     self.gen_num_levels, self.gen_hidden_channels,
                     self.gen_max_hidden_channels)
        model.apply(init_weights)
        return model

    @property
    def discriminator(self) -> nn.Module:
        model = PatchDiscriminator(self.dis_in_channels, self.dis_out_channels,
                                   self.dis_num_levels, self.dis_hidden_channels)
        model.apply(init_weights)
        return model


@dataclass
class TrainConfig:
    wandb_project = "dl_gan_oxford_cats"

    device = "cuda:0"# if torch.cuda.is_available() else "cpu"

    num_epochs = 100

    train_batch = 8
    valid_batch = 512

    num_workers = 0

    dis_loss_coef = 0.02
    min_dis_loss = 0.3
    steps_to_skip = 1
    steps_to_wait = 1

    l1_coef = 1
    edge_coef = 0
    vgg_coef = 0.03

    valid_metric_name = "lpips"

    gen_lr = 2e-4
    dis_lr = 2e-5

    betas = (0.5, 0.999)

    gen_grad_clip_threshold = None
    dis_grad_clip_threshold = 1.

    provide_metric_to_scheduler = False
    n_best_save = 1

    def gen_scheduler(self, optimizer):
        return LinearLR(optimizer, start_factor=1, end_factor=0,
                        total_iters=self.num_epochs // 2,
                        start_epoch=self.num_epochs // 2)

    def dis_scheduler(self, optimizer):
        return LinearLR(optimizer, start_factor=1, end_factor=0,
                        total_iters=self.num_epochs // 2,
                        start_epoch=self.num_epochs // 2)

    @property
    def loss(self) -> dict:
        criterion = dict()
        if self.l1_coef:
            criterion["l1_loss"] = self.l1_coef, L1Loss()
        if self.edge_coef:
            criterion["edge_loss"] = self.edge_coef, EdgeLoss(denoise=True).to(self.device)
        if self.vgg_coef:
            criterion["vgg_perceptual_loss"] = self.vgg_coef, PreprocessWrapper(
                RandomShift(),
                VGGPerceptualLoss(avgpool=True)
            ).to(self.device)
        return criterion

    @property
    def metric(self) -> dict:
        return {"psnr": PeakSignalNoiseRatio().to(self.device),
                "ssim": StructuralSimilarityIndexMeasure().to(self.device),
                "lpips": NegativeLPIPS(verbose=False).to(self.device)}

    @property
    def run_name(self) -> str:
        return f"oxfordcats_disloss{self.dis_loss_coef}_genlr{self.gen_lr}" \
               f"_dislr{self.dis_lr}_batch{self.train_batch}"
