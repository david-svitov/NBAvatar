import torch

class DiceLoss(torch.nn.Module):
    def __init__(self, weight=1.):
        super().__init__()
        self.name = "dice"
        self._weight = weight

    def dice_loss(self, pred, target, smooth=1e-6):
        """
        pred: tensor [N, ...]
        target: tensor [N, ...]
        """
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

        return 1 - dice

    def forward(self, data_dict):
        gt_mask = data_dict["mask"][..., 0:1]
        render_mask = data_dict["billboards_alpha"]

        loss = self.dice_loss(gt_mask, render_mask) * self._weight
        return loss
