import torch


class MSELoss(torch.nn.Module):
    def __init__(self, weight=1., render_source="render_rgb"):
        super().__init__()
        self.name = "mse" + render_source.split('_')[-1]
        self._weight = weight
        self._render_source = render_source
        self._loss_fn = torch.nn.MSELoss()

    def forward(self, data_dict):
        gt_image = data_dict["rgb"]
        render_image = data_dict[self._render_source]

        loss = self._loss_fn(gt_image, render_image) * self._weight
        return loss
