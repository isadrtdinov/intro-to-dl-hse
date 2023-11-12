import torch

from typing import Sequence
from tqdm.auto import tqdm


def predict(model, data_loader, device = "cuda:0", post_process = None):
    """
    Run model on data and collect predictions.

    Args:
        model (torch.nn.Module): model to be tested
        data_loader (torch.utils.data.Dataloader): dataloader with predict set
        device (torch.device): device to train on

    Returns:
        predicted betas
    """

    preds = []
    ppreds = []

    model.eval()
    for pair in tqdm(data_loader, desc="Predict"):
        if isinstance(pair, Sequence):
            input, target = pair
        else:
            input = pair
        input = input.to(device)
        with torch.no_grad():
            pred = model(input)

        preds += [pred.cpu()]

        if post_process is not None:
            ppred = post_process(input, pred)
            ppreds += [ppred.cpu()]

    preds = torch.cat(preds, dim=0)
    if post_process is not None:
        ppreds = torch.cat(ppreds, dim=0)
        return preds, ppreds
    return preds
