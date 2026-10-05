import torch


class OffsetsLoss(torch.nn.Module):
    def __init__(self, weight=1.):
        super().__init__()
        self.name = "offsets"
        self._weight = weight

    def forward(self, data_dict):
        loss = torch.abs(data_dict["xyz_offsets"]).mean() * self._weight
        return loss
