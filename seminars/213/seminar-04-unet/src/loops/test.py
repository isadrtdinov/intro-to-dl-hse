import torch

from collections import defaultdict
from tqdm.auto import tqdm


def test(model, metric, test_loader,
         device = "cuda:0", metric_name = "Metric"):
    """
    Run model on test data and compute metrics.

    Args:
        model (torch.nn.Module): model to be tested
        metric (torch.nn.Module or dict): metric function/s
        test_loader (torch.utils.data.Dataloader): dataloader with test set
        device (torch.device): device to train on
        metric_name (str): metric name if single torch.nn.Module is provided

    Returns:
        tuple of (predictions, targets, test metrics)
    """
    if isinstance(metric, torch.nn.Module):
        metric = {metric_name: metric}

    test_metrics = defaultdict(float)
    model.eval()
    with torch.no_grad():
        for input, target in tqdm(test_loader, desc=f"Test"):
            input = input.to(device)
            target = target.to(device)

            output = model(input)

            for key, m in metric.items():
                test_metrics[key] += m(output, target).item()

    for key in test_metrics:
        test_metrics[key] /= len(test_loader)
        print("Test %s: %.3f" % (key, test_metrics[key]))

    return test_metrics
