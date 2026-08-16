#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from src.registries import LOSS_REGISTRY

class Binary_DiceLoss(nn.Module):
    def __init__(self, 
            batch: Optional[bool] = True, 
            ignore_index: Optional[int] = -1, 
            with_logits: Optional[bool] = True,
            **kwargs,
            )->torch.Tensor:
        super(Binary_DiceLoss, self).__init__()
        self.batch = batch
        self.with_logits = with_logits
        self.ignore_index = ignore_index
        
    def soft_dice_coeff(self, y_true, y_pred):
        # Filter predictions with ignore_index label from loss computation
        not_ignored = y_true != self.ignore_index
        y_pred = y_pred[not_ignored]
        y_true = y_true[not_ignored]
        smooth = 0.00001  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        #score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss
        
    def __call__(self, y_pred, y_true):
        if y_pred.shape != y_true.shape:
            y_pred = y_pred.reshape(y_true.shape)
        if self.with_logits:
            y_pred = torch.sigmoid(y_pred)
        return self.soft_dice_loss(y_true, y_pred)

@LOSS_REGISTRY.register("dice_loss")
class MultiClass_DiceLoss(nn.Module):
    def __init__(self, 
                weight: torch.Tensor, 
                batch: Optional[bool] = True, 
                ignore_index: Optional[int] = -1,
                with_logits: Optional[bool] = True,
                **kwargs,
                )->torch.Tensor:
        super(MultiClass_DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.weight = weight
        self.with_logits = with_logits
        self.binary_diceloss = Binary_DiceLoss(batch, False)

    def __call__(self, y_pred, y_true):
        if self.with_logits:
            y_pred = torch.softmax(y_pred, dim=1)
        if len(y_true.shape) > 3:
            y_true = y_true.squeeze(1)
        # y_true = F.one_hot(y_true.long(), y_pred.shape[1]).permute(0,3,1,2)
        total_loss = 0.0
        tmp_i = 0.0
        for i in range(y_pred.shape[1]):
            if i != self.ignore_index:
                fg = (y_true==i).float()
                diceloss = self.binary_diceloss(y_pred[:, i, :, :], fg)
                total_loss += torch.mul(diceloss, self.weight[i])
                tmp_i += 1.0
        return total_loss / tmp_i

__all__ = [
    "Binary_DiceLoss",
    "MultiClass_DiceLoss"
]


if __name__=="__main__":
    num_class = 10
    logits = torch.rand(size=(2, num_class, 256, 256), dtype=torch.float32)
    target = torch.randint(high=num_class, size=(2, 256, 256))
    # target_onehot = F.one_hot(target, num_class)
    target = torch.tensor(target).long()   #(2, 256, 256)
    mc_dice = MultiClass_DiceLoss(torch.ones(num_class))
    loss = mc_dice(logits, target)
    print(loss)
    logits = torch.rand(size=(2, 1, 256, 256), dtype=torch.float32)
    target = torch.randint(high=2, size=(2, 256, 256))
    # target_onehot = F.one_hot(target, num_class)
    # target = torch.tensor(target).long()   #(2, 256, 256)
    bi_dice = Binary_DiceLoss()
    loss2 = bi_dice(logits, target)
    print(loss2)
