import math

from . import oxford_iiit_cats_edges
from .oxford_iiit_cats_edges import DataConfig, ModelConfig, dataclass


@dataclass
class TrainConfig(oxford_iiit_cats_edges.TrainConfig):
    num_epochs = 200

    dis_loss_coef = 0.
    min_dis_loss = math.inf

    vgg_coef = 0

    dis_lr = 0.

    gen_grad_clip_threshold = None

    @property
    def run_name(self) -> str:
        return f"oxfordcatsedges_super_mae{self.l1_coef}" \
               f"_edge{self.edge_coef}_vgg{self.vgg_coef}" \
               f"_genlr{self.gen_lr}_batch{self.train_batch}"
