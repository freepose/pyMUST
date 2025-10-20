#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn.functional as F

# from torch.nn.functional import one_hot

from .base import AbstractMetric


class CrossEntropy(AbstractMetric):
    """
    Streaming CrossEntropyLoss metric with online aggregation.
    Supports both 'mean' and 'sum' reductions.
    """

    def __init__(self, reduction: str = "mean"):
        assert reduction in ["mean", "sum"], "Only 'mean' or 'sum' supported"
        self.reduction = reduction
        self.reset()

    def reset(self):
        """Clear accumulated statistics."""
        self.total_loss = 0.0
        self.total_samples = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Accumulate batch loss for streaming.
        """
        batch_size = targets.numel()
        if self.reduction == "mean":
            batch_loss = F.cross_entropy(logits, targets, reduction="mean")
            self.total_loss += batch_loss.item() * batch_size
        else:  # reduction == "sum"
            batch_loss = F.cross_entropy(logits, targets, reduction="sum")
            self.total_loss += batch_loss.item()

        self.total_samples += batch_size

    def compute(self) -> float:
        """Return aggregated loss over all updates so far."""
        if self.total_samples == 0:
            return 0.0
        if self.reduction == "mean":
            return self.total_loss / self.total_samples
        else:  # sum
            return self.total_loss

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute non-streaming CrossEntropy loss on a batch."""
        return F.cross_entropy(logits, targets, reduction=self.reduction)


class Accuracy(AbstractMetric):
    """
    Streaming Accuracy metric.
    Supports both binary and multi-class classification.

    Accumulates correct predictions over time.
    """

    def __init__(self, topk: int = 1):
        super().__init__()
        self.topk = topk
        self.reset()

    def reset(self):
        self.correct = torch.tensor(0.0)
        self.total = torch.tensor(0.0)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        if logits.ndim > 1 and logits.size(1) > 1:
            # Multi-class classification
            _, preds = torch.topk(logits, k=self.topk, dim=1)
            correct = preds.eq(targets.unsqueeze(1)).any(dim=1).float().sum()
        else:
            # Binary classification
            preds = (logits > 0).long().view(-1)
            correct = preds.eq(targets.view(-1)).float().sum()

        self.correct += correct
        self.total += targets.numel()

    def compute(self) -> torch.Tensor:
        if self.total == 0:
            return torch.tensor(0.0)
        return self.correct / self.total

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim > 1 and logits.size(1) > 1:
            _, preds = torch.max(logits, dim=1)
        else:
            preds = (logits > 0).long().view(-1)
        return preds.eq(targets.view(-1)).float().mean()


class Precision(AbstractMetric):
    """
    Streaming Precision metric.
    Works for binary or multi-class classification (micro-averaged).
    """

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.tp = torch.tensor(0.0)
        self.fp = torch.tensor(0.0)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        if logits.ndim > 1 and logits.size(1) > 1:
            preds = torch.argmax(logits, dim=1)
        else:
            preds = (logits > 0).long().view(-1)

        preds = preds.view(-1)
        targets = targets.view(-1)

        self.tp += ((preds == 1) & (targets == 1)).sum()
        self.fp += ((preds == 1) & (targets == 0)).sum()

    def compute(self) -> torch.Tensor:
        denom = self.tp + self.fp
        if denom == 0:
            return torch.tensor(0.0)
        return self.tp / denom

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim > 1 and logits.size(1) > 1:
            preds = torch.argmax(logits, dim=1)
        else:
            preds = (logits > 0).long().view(-1)

        preds = preds.view(-1)
        targets = targets.view(-1)

        tp = ((preds == 1) & (targets == 1)).sum()
        fp = ((preds == 1) & (targets == 0)).sum()

        return tp.float() / (tp + fp + 1e-8)


class Recall(AbstractMetric):
    """
    Streaming Recall metric.
    Works for binary or multi-class classification (micro-averaged).
    """

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.tp = torch.tensor(0.0)
        self.fn = torch.tensor(0.0)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        if logits.ndim > 1 and logits.size(1) > 1:
            preds = torch.argmax(logits, dim=1)
        else:
            preds = (logits > 0).long().view(-1)

        preds = preds.view(-1)
        targets = targets.view(-1)

        self.tp += ((preds == 1) & (targets == 1)).sum()
        self.fn += ((preds == 0) & (targets == 1)).sum()

    def compute(self) -> torch.Tensor:
        denom = self.tp + self.fn
        if denom == 0:
            return torch.tensor(0.0)
        return self.tp / denom

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim > 1 and logits.size(1) > 1:
            preds = torch.argmax(logits, dim=1)
        else:
            preds = (logits > 0).long().view(-1)

        preds = preds.view(-1)
        targets = targets.view(-1)

        tp = ((preds == 1) & (targets == 1)).sum()
        fn = ((preds == 0) & (targets == 1)).sum()

        return tp.float() / (tp + fn + 1e-8)
