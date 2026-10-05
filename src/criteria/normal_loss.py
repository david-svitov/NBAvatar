import torch


class NormalLoss(torch.nn.Module):
    def __init__(self, lambda_normal=0.05, lambda_dist=100.):
        super().__init__()
        self.name = "normal"
        self._lambda_normal = lambda_normal
        self._lambda_dist = lambda_dist

    def forward(self, data_dict):
        rend_dist = data_dict["render_dist"]
        rend_normal = data_dict['render_normal']
        surf_normal = data_dict['surf_normal']
        normal_error = 1 - (rend_normal * surf_normal)
        normal_loss = self._lambda_normal * (normal_error).mean()
        dist_loss = self._lambda_dist * (rend_dist).mean()
        return normal_loss + dist_loss
