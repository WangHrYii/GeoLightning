import torch
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F
from src.registries import LOSS_REGISTRY
"""
reference:
        https://github.com/qubvel/segmentation_models.pytorch/tree/a6e1123983548be55d4d1320e0a2f5fd9174d4ac/segmentation_models_pytorch/losses
        https://github.com/catalyst-team/catalyst/blob/master/catalyst/contrib/nn/criterion
"""

@LOSS_REGISTRY.register("ce")
class CrossEntropyLoss(nn.Module):
    def __init__(self, weight: torch.Tensor, 
                    ignore_index: Optional[int] = -1, **kwargs
                    ) -> torch.Tensor:
        """naive cross-entropy loss
        
        Args:
            weight: a torch.Tensor of shape (num_class)
            ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient.
                Default:-1
        """
        super(CrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.weight = weight
        self.ce_loss = nn.CrossEntropyLoss(
            weight=self.weight, ignore_index=self.ignore_index
        )

    def forward(self, scores, labels):  # not for binnary seg
        """
        Shape
             - **scores** - torch.Tensor of shape NxCxHxW
             - **labels** - torch.Tensor of shape NxHxW or NxHxW
        """
        
        if len(labels.shape) > 3:
            labels = labels.squeeze(1)
        # print(scores.shape)
        # print(labels.long().shape)
        loss = self.ce_loss(scores, labels.long())
        return loss

@LOSS_REGISTRY.register("bce_loss")
class BCEWithLogitsLoss(nn.Module):
    def __init__(self, 
        with_logits: Optional[bool] = True, 
        ignore_index: Optional[int] = -1, 
        **kwargs
        )->torch.Tensor:
        super(BCEWithLogitsLoss, self).__init__()
        """naive binary cross-entropy loss
        """
        if with_logits:
            self.bce_loss = nn.BCEWithLogitsLoss()
        else:
            self.bce_loss = nn.BCELoss()
        self.ignore_index = ignore_index

    def forward(self, scores, labels):
        """
        Shape
             - **scores** - torch.Tensor of shape NxHxW or Nx1xHxW
             - **labels** - torch.Tensor of shape NxHxW or Nx1xHxW
        """
        
        if scores.shape != labels.shape:
            scores = scores.reshape(labels.shape)
        not_ignored = labels != self.ignore_index
        scores = scores[not_ignored]
        labels = labels[not_ignored]
        return self.bce_loss(scores, labels.float())

def label_smoothed_nll_loss(
    lprobs: torch.Tensor, target: torch.Tensor, epsilon: float, ignore_index=None, reduction="mean", dim=-1
) -> torch.Tensor:
    """
    Source: https://github.com/pytorch/fairseq/blob/master/fairseq/criterions/label_smoothed_cross_entropy.py
    :param lprobs: Log-probabilities of predictions (e.g after log_softmax)
    :param target:
    :param epsilon:
    :param ignore_index:
    :param reduction:
    :return:
    """
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(dim)

    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        target = target.masked_fill(pad_mask, 0)
        nll_loss = -lprobs.gather(dim=dim, index=target.long())
        smooth_loss = -lprobs.sum(dim=dim, keepdim=True)

        # nll_loss.masked_fill_(pad_mask, 0.0)
        # smooth_loss.masked_fill_(pad_mask, 0.0)
        nll_loss = nll_loss.masked_fill(pad_mask, 0.0)
        smooth_loss = smooth_loss.masked_fill(pad_mask, 0.0)
    else:
        nll_loss = -lprobs.gather(dim=dim, index=target)
        smooth_loss = -lprobs.sum(dim=dim, keepdim=True)

        nll_loss = nll_loss.squeeze(dim)
        smooth_loss = smooth_loss.squeeze(dim)

    if reduction == "sum":
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    if reduction == "mean":
        nll_loss = nll_loss.mean()
        smooth_loss = smooth_loss.mean()

    eps_i = epsilon / lprobs.size(dim)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss


class SoftCrossEntropyLoss(nn.Module):

    __constants__ = ["reduction", "ignore_index", "smooth_factor"]

    def __init__(
        self,
        reduction: str = "mean",
        smooth_factor: Optional[float] = 0.1,
        ignore_index: Optional[int] = -100,
        dim: int = 1, **kwargs
    ):
        """Drop-in replacement for torch.nn.CrossEntropyLoss with label_smoothing
        
        Args:
            smooth_factor: Factor to smooth target (e.g. if smooth_factor=0.1 then [1, 0, 0] -> [0.9, 0.05, 0.05])
        
        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W)
        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        super().__init__()
        self.smooth_factor = smooth_factor
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.dim = dim

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(y_pred, dim=self.dim)
        return label_smoothed_nll_loss(
            log_prob,
            y_true,
            epsilon=self.smooth_factor,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            dim=self.dim,
        )


class SoftBCEWithLogitsLoss(nn.Module):

    __constants__ = ["weight", "pos_weight", "reduction", "ignore_index", "smooth_factor"]

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        ignore_index: Optional[int] = None,
        reduction: str = "mean",
        smooth_factor: Optional[float] = 0.1,
        pos_weight: Optional[torch.Tensor] = None, 
        with_logits: Optional[bool] = True, **kwargs
    )->torch.Tensor:
        """Drop-in replacement for torch.nn.BCEWithLogitsLoss with few additions: ignore_index and label_smoothing
        
        Args:
            ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient. 
            smooth_factor: Factor to smooth target (e.g. if smooth_factor=0.1 then [1, 0, 1] -> [0.9, 0.1, 0.9])
        
        Shape
             - **y_pred** - torch.Tensor of shape NxCxHxW
             - **y_true** - torch.Tensor of shape NxHxW or Nx1xHxW
        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.smooth_factor = smooth_factor
        self.register_buffer("weight", weight)
        self.register_buffer("pos_weight", pos_weight)
        if with_logits:
            self.bce_loss = nn.BCEWithLogitsLoss(reduction = 'none')
        else:
            self.bce_loss = nn.BCELoss(reduction = 'none')

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: torch.Tensor of shape (N, C, H, W)
            y_true: torch.Tensor of shape (N, H, W)  or (N, 1, H, W)
        
        Returns:
            loss: torch.Tensor
        """

        if y_pred.shape != y_true.shape:
            y_true = y_true.reshape(y_pred.shape)
        if self.smooth_factor is not None:
            soft_targets = (1 - y_true) * self.smooth_factor + y_true * (1 - self.smooth_factor)
        else:
            soft_targets = y_true.float()
        loss = self.bce_loss(
            y_pred, soft_targets,
        )

        if self.ignore_index is not None:
            not_ignored_mask = y_true != self.ignore_index
            loss *= not_ignored_mask.type_as(loss)

        if self.reduction == "mean":
            loss = loss.mean()

        if self.reduction == "sum":
            loss = loss.sum()

        return loss


class SymmetricCrossEntropyLoss(nn.Module):
    """The Symmetric Cross Entropy loss.
    It has been proposed in `Symmetric Cross Entropy for Robust Learning
    with Noisy Labels`_.
    .. _Symmetric Cross Entropy for Robust Learning with Noisy Labels:
        https://arxiv.org/abs/1908.06112
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, **kwargs):
        """
        Args:
            alpha(float):
                corresponds to overfitting issue of CE
            beta(float):
                corresponds to flexible exploration on the robustness of RCE
        """
        super(SymmetricCrossEntropyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, input_: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculates loss between ``input_`` and ``target`` tensors.
        Args:
            input_: input tensor of size
                (batch_size, num_classes)
            target: target tensor of size (batch_size), where
                values of a vector correspond to class index
        Returns:
            torch.Tensor: computed loss
        """
        num_classes = input_.shape[1]
        target_one_hot = F.one_hot(target, num_classes).float()
        assert target_one_hot.shape == input_.shape

        input_ = torch.clamp(input_, min=1e-7, max=1.0)
        target_one_hot = torch.clamp(target_one_hot, min=1e-4, max=1.0)

        cross_entropy = (-torch.sum(target_one_hot * torch.log(input_), dim=1)).mean()
        reverse_cross_entropy = (-torch.sum(input_ * torch.log(target_one_hot), dim=1)).mean()
        loss = self.alpha * cross_entropy + self.beta * reverse_cross_entropy
        return loss


class MaskCrossEntropyLoss(nn.Module):
    """@TODO: Docs. Contribution is welcome."""

    def __init__(self, *args, **kwargs):
        """@TODO: Docs. Contribution is welcome."""
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(*args, **kwargs, reduction="none")

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculates loss between ``logits`` and ``target`` tensors.
        Args:
            logits: model logits
            target: true targets
            mask: targets mask
        Returns:
            torch.Tensor: computed loss
        """
        loss = self.ce_loss.forward(logits, target)
        loss = torch.mean(loss[mask == 1])
        return loss


class BatchCrossEntropyLoss(nn.Module):
    def __init__(self,
                    reduction: Optional[str] = 'mean',
                    ignore_index: Optional[int] = -1, **kwargs
                    ) -> torch.Tensor:
        """naive cross-entropy loss
        
        Args:
            weight: a torch.Tensor of shape (num_class)
            ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient.
                Default:-1
        """
        super(BatchCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction=self.reduction)

    def forward(self, scores, labels):  # not for binnary seg
        """
        Shape
             - **scores** - torch.Tensor of shape NxCxHxW
             - **labels** - torch.Tensor of shape NxHxW or NxHxW
        """
        if len(labels.shape) > 3:
            labels = labels.squeeze(1)
        loss = self.ce_loss(scores, labels.long())
        return loss

class BatchBCEWithLogitsLoss(nn.Module):
    def __init__(self, 
        with_logits: Optional[bool] = True, 
        ignore_index: Optional[int] = -1, 
        **kwargs
        )->torch.Tensor:
        super(BatchBCEWithLogitsLoss, self).__init__()
        """naive binary cross-entropy loss
        """
        if with_logits:
            self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        else:
            self.bce_loss = nn.BCELoss(reduction='none')
        self.ignore_index = ignore_index

    def forward(self, scores, labels):
        """
        Shape
             - **scores** - torch.Tensor of shape NxHxW or Nx1xHxW
             - **labels** - torch.Tensor of shape NxHxW or Nx1xHxW
        """
        
        if scores.shape != labels.shape:
            scores = scores.reshape(labels.shape)
        not_ignored = labels != self.ignore_index
        scores = scores[not_ignored]
        labels = labels[not_ignored]
        return self.bce_loss(scores, labels.float())


__all__ = [
    "MaskCrossEntropyLoss",
    "SymmetricCrossEntropyLoss",
    "CrossEntropyLoss",
    "BCEWithLogitsLoss",
    "SoftCrossEntropyLoss",
    "SoftBCEWithLogitsLoss",
    "BatchCrossEntropyLoss",
    "BatchBCEWithLogitsLoss",
]

if __name__=="__main__":
    num_class = 10
    logits = torch.rand(size=(2, num_class, 256, 256), dtype=torch.float32)
    target = torch.randint(high=num_class-1, size=(2, 256, 256))
    target_onehot = F.one_hot(target, num_class)
    target = torch.tensor(target).long()   #(2, 256, 256)
    naive_ce = CrossEntropyLoss(torch.ones(num_class))
    loss = naive_ce(logits, target)
    print(loss)
