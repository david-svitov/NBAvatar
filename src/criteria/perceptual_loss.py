import random

import torch
from lpips import lpips


class PerceptualLoss(torch.nn.Module):
    def __init__(self, weight=1., random_patch_from=40_000, start_from=0, render_source="render_rgb"):
        super().__init__()
        self.name = "lpips" + render_source.split('_')[-1]
        self._weight = weight
        self._random_patch_from = random_patch_from
        self._start_from = start_from
        self._render_source = render_source
        self._loss_fn = lpips.LPIPS(net="vgg")

    def forward(self, data_dict):
        if data_dict["global_step"] < self._start_from:
            return torch.tensor(0.)
        self._loss_fn.to(data_dict[self._render_source].device)
        synth_images = torch.permute(data_dict[self._render_source], (0, 3, 1, 2))
        target_images = torch.permute(data_dict["rgb"], (0, 3, 1, 2))

        sub_synth = []
        sub_target = []
        points_proj = torch.cat([data_dict["head_keys_proj"], data_dict["rh_keys_proj"]], dim=1)
        for image_id in range(len(points_proj)):
            if data_dict["global_step"] > self._random_patch_from:
                verts_projected = points_proj[image_id]
                point_on_model = random.choice(verts_projected)
                cx, cy = point_on_model
                h, w = synth_images.shape[2:]
                cx = cx * w
                cy = cy * h
                j = max(int(cx), 128)
                i = max(int(cy), 128)

                i = min(i, synth_images.shape[2] - 129)
                j = min(j, synth_images.shape[3] - 129)

                x_slice = slice(j - 128, j + 128)
                y_slice = slice(i - 128, i + 128)
                sub_synth.append(synth_images[image_id, :, y_slice, x_slice])
                sub_target.append(target_images[image_id, :, y_slice, x_slice])
            else:
                sub_synth.append(synth_images[image_id])
                sub_target.append(target_images[image_id])

        sub_synth = torch.stack(sub_synth, dim=0)
        sub_target = torch.stack(sub_target, dim=0)

        loss = self._loss_fn(sub_target, sub_synth).mean() * self._weight
        return loss
