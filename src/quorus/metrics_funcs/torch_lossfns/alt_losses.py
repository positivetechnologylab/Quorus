import torch
import torch.nn as nn

class QuantumCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss for models that output *probabilities* (e.g., variational quantum circuits),
    not logits.

    - input: probs of shape [batch_size, num_classes], each row should sum to 1
    - target:
        * either class indices of shape [batch_size] (LongTensor),
        * or one-hot / soft labels of shape [batch_size, num_classes].
    """
    def __init__(self, reduction: str = "mean", eps: float = 1e-8):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Invalid reduction: {reduction}")
        self.reduction = reduction
        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        input: probability tensor from your quantum circuit, shape [N, C]
        target: labels, either [N] (class indices) or [N, C] (one-hot / soft labels)
        """
        # Ensure numerical stability: avoid log(0)
        probs = input.clamp(min=self.eps, max=1.0)

        # Case 1: target is one-hot / soft labels (same shape as probs, like BCELoss)
        if target.dim() == probs.dim():
            # Convert target to same dtype as probs (e.g., float32)
            target = target.to(probs.dtype)

            # cross-entropy: - sum_j y_j * log p_j
            log_probs = probs.log()
            per_sample_loss = -(target * log_probs).sum(dim=1)

        # Case 2: target is class indices [N]
        else:
            if target.dtype != torch.long:
                target = target.long()

            log_probs = probs.log()
            # pick log p_true_class for each sample
            per_sample_loss = -log_probs.gather(dim=1, index=target.unsqueeze(1)).squeeze(1)

        if self.reduction == "mean":
            return per_sample_loss.mean()
        elif self.reduction == "sum":
            return per_sample_loss.sum()
        else:  # "none"
            return per_sample_loss