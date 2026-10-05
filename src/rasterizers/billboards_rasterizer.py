import torch
from diff_bbsplat_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from src.utils.train_utils import shift_image, shift_image_pad
from src.utils.util import depth_to_normal


class BillboardsRasterizer(torch.nn.Module):
    def __init__(self, render_size=1080):
        super().__init__()

        self._render_size = render_size

    def forward(self, data_dict, roll_offset=True):
        rendered_image_list = []
        alpha_list = []
        impact_list = []
        surf_normal_list = []
        render_normal_list = []
        render_dist_list = []

        means2D = torch.zeros_like(
            data_dict["billboards_xyz"],
            dtype=data_dict["billboards_xyz"].dtype,
            requires_grad=True,
            device=data_dict["billboards_xyz"].device,
        )
        means2D.retain_grad()
        for frame_id in range(len(data_dict["billboards_xyz"])):
            xyz = data_dict["billboards_xyz"][frame_id]
            texture_alpha = data_dict["texture_alpha"][frame_id]
            texture_color = data_dict["texture_color"][frame_id]
            shs = data_dict["billboards_colors"][frame_id]

            scales = data_dict["billboards_scales"][frame_id]
            rotations = data_dict["billboards_rotations"][frame_id]
            background = data_dict["background"][frame_id]

            projection_matrix = data_dict["camera"]["projection_mat"][frame_id].clone()
            '''
            shift_x_pixels = (projection_matrix[0, 2].item() + 1) * self._render_size / 2 - self._render_size / 2
            shift_y_pixels = (projection_matrix[1, 2].item() + 1) * self._render_size / 2 - self._render_size / 2
            projection_matrix[0, 2] = projection_matrix[1, 2] = 0
            '''
            viewmatrix = torch.eye(4).to(projection_matrix.device)

            means3D = xyz

            full_proj_transform = (viewmatrix.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
            camera_center = viewmatrix.inverse()[3, :3]
            #depths_condition = data_dict["depth_mesh"][frame_id]
            INF_FAR = 1000
            depths_condition = torch.ones_like(data_dict["mask"][frame_id]) * INF_FAR #(1 - data_dict["mask_mesh"][frame_id]) * INF_FAR + depths_condition
            depths_condition = depths_condition.detach()

            tanfovx = 1 / projection_matrix[0, 0]
            tanfovy = 1 / projection_matrix[1, 1]

            p02 = projection_matrix[0, 2].float()
            p12 = projection_matrix[1, 2].float()

            means3D_corr = means3D.clone()
            z = means3D_corr[:, 2]

            means3D_corr[:, 0] = means3D_corr[:, 0] + p02 * z * tanfovx
            means3D_corr[:, 1] = means3D_corr[:, 1] + p12 * z * tanfovy

            raster_settings = GaussianRasterizationSettings(
                image_height=self._render_size,
                image_width=self._render_size,
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=background,
                scale_modifier=1.0,
                viewmatrix=viewmatrix,
                projmatrix=full_proj_transform,
                sh_degree=0,
                campos=camera_center,
                prefiltered=False,
                debug=False,
            )
            rasterizer = GaussianRasterizer(raster_settings=raster_settings)

            rendered_image, radii, impact, allmap = rasterizer(
                means3D=means3D_corr.float(),
                means2D=means2D[frame_id].float(),
                texture_alpha=texture_alpha.float(),
                texture_color=texture_color.float(),
                shs=shs.float(),
                depths_condition=depths_condition.float(),
                scales=scales.float(),
                rotations=rotations.float(),
                cov3D_precomp=None,
            )
            render_alpha = allmap[1:2]
            render_normal = allmap[2:5]  # (3, H, W) view-space normals from rasterizer

            '''
            if roll_offset:
                rendered_image = shift_image(rendered_image, shift_x_pixels, shift_y_pixels)
                render_alpha = shift_image(render_alpha, shift_x_pixels, shift_y_pixels)
            else:
                rendered_image = shift_image_pad(rendered_image, shift_x_pixels, shift_y_pixels)
                render_alpha = shift_image_pad(render_alpha, shift_x_pixels, shift_y_pixels)
            '''

            rendered_image_list.append(rendered_image)
            alpha_list.append(render_alpha)
            impact_list.append(impact)
            render_normal_list.append(render_normal)

        impact = torch.stack(impact_list, dim=0)

        return {
            "billboards_rasterization": torch.stack(rendered_image_list, dim=0),
            "billboards_alpha": torch.stack(alpha_list, dim=0),
            "billboards_normal": torch.stack(render_normal_list, dim=0),
            "impact": impact,
            "viewspace_points": means2D,
            "visibility_filter": impact > 0
        }
