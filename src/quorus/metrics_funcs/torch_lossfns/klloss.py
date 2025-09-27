"""### Custom KLLoss Function Module"""

import torch.nn as nn
import torch
from quorus.logging.custom_slog import print_cust

# copied from depthFL impl, https://github.com/adap/flower/blob/main/baselines/depthfl/depthfl/models.py
class KLLoss(nn.Module):
    """KL divergence loss for self distillation."""

    def __init__(self):
        super().__init__()
        self.temperature = 1

    def forward(self, pred, label):
        """
        Function that computes the KLLoss between the prediction and label.

        Parameters:
            pred (torch.tensor): the predicted value
            label (torch.tensor): the target label

        Returns:
            the KLLoss between the prediction and target label.
        """
        # predict = F.log_softmax(pred / self.temperature, dim=1)
        # target_data = F.softmax(label / self.temperature, dim=1)
        predict = pred.log()
        target_data = label
        target_data = target_data + 10 ** (-7)
        with torch.no_grad():
            target = target_data.detach().clone()

        loss = (
            self.temperature
            * self.temperature
            * ((target * (target.log() - predict)).sum(1).sum() / target.size()[0])
        )
        print_cust(f"KLLoss, forward, loss: {loss}")
        return loss