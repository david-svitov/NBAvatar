import torch
from pytorch3d.ops import knn_points


class KNNReg(torch.nn.Module):
    def __init__(self, K=3, weight=1.):
        super().__init__()
        self.name = "knn"
        self._weight = weight
        self._K = K

    def forward(self, data_dict):
        batch_size = len(data_dict["billboards_xyz"])
        loss = 0
        for i in range(batch_size):
            xyz = data_dict["billboards_xyz"][i]
            scales = data_dict["billboards_scales"][i]
            rotations = data_dict["billboards_rotations"][i]

            dist_sq, nn_ind, _ = knn_points(xyz[None], xyz[None], K=self._K, return_nn=False)
            nn_ind = nn_ind.squeeze(0)
            # reg the std inside knn
            q = rotations[nn_ind, :]  # N,K,4
            q_std = q.std(dim=1).mean()

            s = scales[nn_ind, :]  # N,K,2
            s_std = s.std(dim=1).mean()

            loss += q_std + s_std + dist_sq.mean()

        loss = loss / batch_size  * self._weight
        return loss


