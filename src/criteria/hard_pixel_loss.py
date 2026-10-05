import torch


class HardPixelMSELoss(torch.nn.Module):
    """MSE loss computed only on the top-percentile hardest pixels."""
    def __init__(self, weight=1., percentile=80, render_source="render_rgb"):
        super().__init__()
        self.name = "hard_pixel"
        self._weight = weight
        self._percentile = percentile / 100.0
        self._render_source = render_source

    def forward(self, data_dict):
        gt = data_dict["rgb"]
        render = data_dict[self._render_source]

        per_pixel = ((gt - render) ** 2).mean(dim=-1, keepdim=True)   # (B, H, W, 1)
        threshold = torch.quantile(per_pixel.reshape(-1), self._percentile)
        hard_mask = (per_pixel >= threshold).float()
        loss = (per_pixel * hard_mask).sum() / (hard_mask.sum() + 1e-8)
        return loss * self._weight
