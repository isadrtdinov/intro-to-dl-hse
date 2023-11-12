import torch
import numpy as np
import wandb

from torch import nn, Tensor
from typing import Dict
from collections import defaultdict
from tqdm.auto import tqdm

from ..utils import checkpoint, image_grid


def get_n_best_metric(metrics, n_best):
    return np.partition(metrics[:-1], -n_best)[-n_best]


def losses(generator: nn.Module, discriminator: nn.Module,
           input: Tensor, target: Tensor, criterion: Dict,
           dis_loss_coef: float, skip_dis_step: bool = True):
    # Generator forward step
    output = generator(input)

    # Conditional discriminator inputs
    concat_output = torch.cat((output, input), 1)
    concat_target = torch.cat((target, input), 1)

    # E step
    discriminator.requires_grad(False)

    # Supervised loss
    super_losses = {key: (coef, l(output, target)) for key, (coef, l) in criterion.items()}

    # LSGAN generator loss
    gen_loss = sum(map(lambda x: x[0] * x[1], super_losses.values())) + dis_loss_coef * \
               ((discriminator(concat_output) - 1) ** 2).mean()  # Classify fake as 1 (real)

    # D step
    concat_output = concat_output.detach()

    if not skip_dis_step:
        discriminator.requires_grad(True)

    # LSGAN discriminator loss
    dis_loss = (discriminator(concat_output) ** 2 +
                (discriminator(concat_target) - 1) ** 2).mean()  # Classify fake as 0, real as 1

    return output, super_losses, gen_loss, dis_loss


def train(config, train_loader, valid_loader,
          generator, discriminator, gen_optimizer, dis_optimizer,
          gen_scheduler = None, dis_scheduler = None, resume = False, progress = "epochs"):
    """
    Train model.

    Args:
        config (TrainConfig): config class containing all necessary training parameters
        train_loader (torch.utils.data.Dataloader): dataloader for train set
        valid_loader (torch.utils.data.Dataloader): dataloader for valid set
        generator (nn.Module): model to be fitted
        discriminator (nn.Module): discriminator model
        gen_optimizer (torch.optim.Optimizer): model optimizer
        dis_optimizer (torch.optim.Optimizer): discriminator optimizer
        gen_scheduler (torch.optim.lr_scheduler.): generator optimizer scheduler
        dis_scheduler (torch.optim.lr_scheduler.): discriminator optimizer scheduler
        resume (bool): resume training from the last checkpoint
        progress (str): {"epochs", "samples"} how to draw progressbars

    Returns:
        tuple of (best train metric, best valid metric, best valid metric epoch)

    """
    criterion = config.loss
    metric = config.metric

    if isinstance(criterion, nn.Module):
        criterion = {config.loss_name: criterion}
    if isinstance(metric, nn.Module):
        metric = {config.valid_metric_name: metric}

    if resume:
        start_epoch, valid_metric_history = checkpoint.load(
            config.run_name, generator, gen_optimizer, gen_scheduler,
            discriminator, dis_optimizer, dis_scheduler
        )
    else:
        start_epoch = 0
        valid_metric_history = []

    dis_loss = 1  # Init value for use in stepper
    skip_dis_step = False
    steps_skipped = 0
    steps_waited = 0

    epoch_iter = range(start_epoch + 1, config.num_epochs + 1)
    if progress == "epochs":
        epoch_iter = tqdm(epoch_iter, desc="Epoch")

    for epoch in epoch_iter:
        # Train
        generator.train()
        discriminator.train()

        train_iter = train_loader
        if progress == "samples":
            train_iter = tqdm(train_iter, desc=f"Train {epoch}/{config.num_epochs}")

        for input, target in train_iter:
            input = input.to(config.device)
            target = target.to(config.device)

            # Zero grad
            gen_optimizer.zero_grad()
            dis_optimizer.zero_grad()

            # Skip discriminator update if the loss is too low
            if dis_loss < config.min_dis_loss:
                if not skip_dis_step:
                    steps_waited += 1
                if steps_waited >= config.steps_to_wait:
                    skip_dis_step = True
                    steps_skipped = 0
            elif skip_dis_step:
                steps_skipped += 1
                if steps_skipped >= config.steps_to_skip:
                    skip_dis_step = False
                    steps_waited = 0

            output, super_losses, gen_loss, dis_loss = losses(
                generator, discriminator, input, target,
                criterion, config.dis_loss_coef, skip_dis_step
            )

            if config.gen_grad_clip_threshold is not None:
                nn.utils.clip_grad_norm_(generator.parameters(),
                                         config.gen_grad_clip_threshold)
            if config.dis_grad_clip_threshold is not None:
                nn.utils.clip_grad_norm_(discriminator.parameters(),
                                         config.dis_grad_clip_threshold)

            if (torch.isnan(gen_loss) or torch.isinf(gen_loss) or
                    torch.isnan(dis_loss) or torch.isinf(dis_loss)):
                return "gradient explosion"

            # Generator update
            gen_loss.backward()
            gen_optimizer.step()

            # Discriminator update
            if not skip_dis_step:
                dis_loss.backward()
                dis_optimizer.step()

            # Logger
            log = {"train_" + key: value.item() for key, (_, value) in super_losses.items()}
            log["train_generator_loss"] = gen_loss.item()
            log["train_discriminator_loss"] = dis_loss.item()
            log["discriminator_enabled"] = int(not skip_dis_step)
            with torch.no_grad():
                for key, m in metric.items():
                    log["train_" + key] = m(output, target).item()

            wandb.log(log)

        # Valid
        valid_log = defaultdict(float)
        generator.eval()
        discriminator.eval()

        valid_iter = valid_loader
        if progress == "samples":
            valid_iter = tqdm(valid_iter, desc=f"Valid {epoch}/{config.num_epochs}")

        with torch.no_grad():
            for input, target in valid_iter:
                input = input.to(config.device)
                target = target.to(config.device)

                output, super_losses, gen_loss, dis_loss = losses(
                    generator, discriminator, input, target,
                    criterion, config.dis_loss_coef
                )

                # Logger
                valid_log["valid_generator_loss"] += gen_loss.item()
                valid_log["valid_discriminator_loss"] += dis_loss.item()
                for key, (_, value) in super_losses.items():
                    valid_log["valid_" + key] += value.item()
                for key, m in metric.items():
                    valid_log["valid_" + key] += m(output, target).item()

        for key in valid_log:
            valid_log[key] /= len(valid_loader)

        valid_log = dict(valid_log)
        valid_log["image_samples"] = wandb.Image(image_grid(input, output, target, num_images=4))
        valid_log["generator_lr"] = gen_optimizer.param_groups[0]["lr"]
        valid_log["discriminator_lr"] = dis_optimizer.param_groups[0]["lr"]
        valid_log["epoch"] = epoch

        wandb.log(valid_log, commit=False)

        # Schedulers step
        valid_metric_history += [valid_log["valid_" + config.valid_metric_name]]

        if config.provide_metric_to_scheduler:
            if gen_scheduler is not None:
                gen_scheduler.step(valid_metric_history[-1])
            if dis_scheduler is not None:
                dis_scheduler.step(valid_metric_history[-1])
        else:
            if gen_scheduler is not None:
                gen_scheduler.step()
            if dis_scheduler is not None:
                dis_scheduler.step()

        # Save model if higher metric is achieved
        if (epoch <= config.n_best_save or valid_metric_history[-1] >
                get_n_best_metric(valid_metric_history, config.n_best_save)):
            checkpoint.save_model(config.run_name, epoch, generator)

        # Save full checkpoint
        checkpoint.save(config.run_name, epoch, valid_metric_history,
                        generator, gen_optimizer, gen_scheduler,
                        discriminator, dis_optimizer, dis_scheduler)

    print("Best valid %s:  %.3f on epoch %d" %
          (config.valid_metric_name, max(valid_metric_history), np.argmax(valid_metric_history) + 1))

    return "success"
