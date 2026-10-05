import copy
import os
from array import array

import cv2
import mano
import numpy as np
import tinyobjloader
import torch
import torch.optim as optim
from pytorch3d.transforms import matrix_to_quaternion, quaternion_multiply, quaternion_apply
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.rasterizers.billboards_rasterizer import BillboardsRasterizer
from src.utils import transformation_util as tranu, tracking_util, FLAME_util
from src.utils.spectral_utils import spectral_face_channels
from src.utils.train_utils import dict2device
from src.utils.unet_utils import UNetSmall
from src.utils.util import RGB2SH, inverse_sigmoid, s_inv_act, s_act, q_normalize, instantiate_from_config


def _point_to_bary(P, V0, V1, V2):
    """Barycentric coords of P in triangle (V0,V1,V2): returns (u,v,w) s.t. u*V0+v*V1+w*V2≈P."""
    e0, e1 = V2 - V0, V1 - V0
    e2 = P - V0
    d00, d01, d11 = np.dot(e0, e0), np.dot(e0, e1), np.dot(e1, e1)
    d02, d12 = np.dot(e0, e2), np.dot(e1, e2)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-10:
        return 1 / 3., 1 / 3., 1 / 3.
    inv = 1.0 / denom
    w = (d11 * d02 - d01 * d12) * inv  # coefficient for V2
    v = (d00 * d12 - d01 * d02) * inv  # coefficient for V1
    return 1.0 - v - w, v, w           # (u, v, w) for (V0, V1, V2)


class Trainer(nn.Module):

    def setup_functions(self):
        self.scaling_activation = s_act
        self.scaling_inverse_activation = s_inv_act

        self.opacity_activation = torch.sigmoid
        self.color_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.inverse_color_activation = inverse_sigmoid
        self.rotation_activation = q_normalize

    def __init__(self, model_path, criteria, render_size=1080, device="cuda", n_pca=45,
                    num_iterations=100_000, log_iteration=10_000, batch_size=1, test_mode=False,
                    n_expression=50, n_head_pose=6, triangle_walk=False, billboards_per_face=1,
                    num_channels=6, texture_size=None, use_normals=False,
                    mse_finetune_steps=0, mse_finetune_lr=1e-4, mse_soup_alpha=0.8,
                    finetune_criteria=None, use_renderer=True, use_deformations=True):
        super().__init__()
        self.setup_functions()

        self.num_channels = num_channels
        self.device = device
        self._min_s_value = 0.0
        self._max_s_value = 0.1
        self._billboards_is_active = False
        self._texture_color_lr = 0.0005
        self._texture_opacity_lr = 0.0005

        self._use_normals = use_normals
        self._use_renderer = use_renderer
        self._use_deformations = use_deformations
        self._mse_finetune_steps = mse_finetune_steps
        self._mse_finetune_lr = mse_finetune_lr
        self._mse_soup_alpha = mse_soup_alpha

        self.num_iterations = num_iterations
        self.log_iteration = log_iteration
        self.batch_size = batch_size
        self.log_folder = "./logs"

        if not test_mode:
            self._setup_criteria(criteria)
            self._setup_finetune_criteria(finetune_criteria)

        flame_model, flame_faces, rh_model, rh_faces = self._load_models(model_path, n_pca)
        self._flame_model = flame_model
        self._flame_faces = flame_faces
        self._rh_model = rh_model
        self._rh_faces = rh_faces

        rh_vt, rh_ft = self._load_uv_maps(model_path, "mano_v1_2/models/MANO_UV_right.obj")
        head_vt, head_ft = self._load_uv_maps(model_path, "head_template_mesh.obj")

        head_vt[:, 0] = head_vt[:, 0] / 2.
        rh_vt[:, 0] = 0.5 + rh_vt[:, 0] / 2.

        self._uv = np.concatenate([head_vt, rh_vt], 0)
        self._uv_idx = np.concatenate([head_ft, rh_ft + len(head_vt)], 0)
        self._uv = torch.from_numpy(self._uv).to(self.device)
        self._uv_idx = torch.from_numpy(self._uv_idx).to(self.device)

        self._render_size = render_size

        image = cv2.imread("assets/alpha_init_gaussian_small.png")[..., 0] / 255.0
        if texture_size is not None:
            image = cv2.resize(image, (texture_size, texture_size))
        self._texture_alpha_init = torch.tensor([image], dtype=torch.float, device="cuda")

        number_of_faces = len(flame_faces) + len(rh_faces)
        number_of_points = number_of_faces * billboards_per_face
        self._change_parameters_shape(number_of_points, billboards_per_face)
        self._precompute_face_adjacency()
        self._triangle_walk_interval = 100

        self._triangle_walk = triangle_walk
        self._billboards_rasterizer = BillboardsRasterizer(render_size)
        # pose_dim: head (global_orient 3 + transl 3 + expression n_expression + pose n_head_pose)
        #         + hand (global_orient 3 + transl 3 + hand_pose n_pca)
        pose_dim = 3 + 3 + n_expression + n_head_pose + 3 + 3 + n_pca
        renderer_input_channels = self.num_channels
        if use_normals:
            renderer_input_channels += 3
        self._renderer = UNetSmall(renderer_input_channels, pose_dim) if use_renderer else nn.Identity()

    @property
    def use_renderer(self):
        return self._use_renderer

    def set_log_folder(self, log_folder):
        self.log_folder = log_folder

    def _setup_criteria(self, criteria_config):
        self._criteria = {}
        if criteria_config is None:
            return
        for name, config in criteria_config.items():
            criterion = instantiate_from_config(config)
            self._criteria[name] = criterion

    def _setup_finetune_criteria(self, criteria_config):
        self._finetune_criteria = {}
        if criteria_config is None:
            return
        for name, config in criteria_config.items():
            criterion = instantiate_from_config(config)
            self._finetune_criteria[name] = criterion

    def _load_models(self, model_path, n_pca):
        asset_path = os.path.join(model_path, "assets")
        mano_model_path = asset_path + '/mano_v1_2/models/MANO_RIGHT.pkl'
        flame_model_path = asset_path + "/generic_model.pkl"
        flame_landmark_path = asset_path + "/landmark_embedding.npy"

        flame_model = FLAME_util.FLAME(
            flame_model_path,
            flame_landmark_path).to(self.device)
        flame_faces = FLAME_util.get_FLAME_faces(flame_model_path)
        flame_faces = flame_faces.to(self.device).to(torch.int32)

        rh_model = mano.model.load(
            model_path=mano_model_path,
            is_right=True,
            num_pca_comps=n_pca,
            batch_size=1,
            flat_hand_mean=True).to(self.device)
        rh_faces = torch.from_numpy(rh_model.faces.astype(np.int32)).to(self.device)

        return flame_model, flame_faces, rh_model, rh_faces

    def _read_ft(self, shapes):
        texcoord_indices = array('i')

        for shape in shapes:
            indices = shape.mesh.indices
            for i in range(len(indices)):
                texcoord_indices.append(indices[i].texcoord_index)

        ft_flat = np.array(texcoord_indices, dtype=np.int32)
        return ft_flat.reshape(-1, 3)

    def _load_uv_maps(self, model_path, obj_filename):
        asset_path = os.path.join(model_path, "assets")
        mano_obj_path = os.path.join(asset_path, obj_filename)

        reader = tinyobjloader.ObjReader()
        reader.ParseFromFile(mano_obj_path)
        attrib = reader.GetAttrib()
        shapes = reader.GetShapes()

        vt = np.array(attrib.texcoords, dtype=np.float32).reshape(-1, 2)
        ft = self._read_ft(shapes)
        return vt, ft

    def _change_parameters_shape(self, npoints, billboards_per_face=1):
        self._xyz = torch.nn.Parameter(torch.zeros([npoints, 3], device=self.device), requires_grad=True)
        self._color = torch.nn.Parameter(RGB2SH(torch.ones([npoints, 3, 3], device=self.device) * 0.5),
                                         requires_grad=True)
        self._rotation = torch.nn.Parameter(torch.zeros([npoints, 4], device=self.device), requires_grad=False)
        self._rotation[:, 0] = 0.71
        self._rotation[:, 1] = 0.71
        self._rotation.requires_grad = True
        self._scaling = torch.nn.Parameter(
            self.scaling_inverse_activation(torch.ones([npoints, 2], device=self.device) * 0.07, self._min_s_value,
                                            self._max_s_value),
            requires_grad=True)

        nfaces = npoints // billboards_per_face
        face_indices = torch.arange(nfaces, dtype=torch.long, device=self.device).repeat_interleave(billboards_per_face)
        self._billboard_to_face = torch.nn.Parameter(face_indices, requires_grad=False)
        self._max_impact = torch.nn.Parameter(torch.zeros([npoints], device=self.device), requires_grad=False)

        # Drifting attachment (SplattingAvatar-style):
        # _bary: current barycentric coords (u, v, w) for each billboard, updated by walking
        # _xyz: (Δu, Δv, d) — gradient-driven barycentric deltas + normal displacement
        bary = torch.ones([npoints, 3], device=self.device) / 3.0
        if billboards_per_face > 1:
            bary += torch.randn_like(bary) * 0.05
            bary = bary.abs()
            bary = bary / bary.sum(dim=-1, keepdim=True)
        self._bary = nn.Parameter(bary, requires_grad=False)

        # Setup trainable alpha textures
        texture_alpha = self._texture_alpha_init.repeat(npoints, 1, 1)
        texture_alpha = self.inverse_opacity_activation(texture_alpha)
        self._texture_alpha = nn.Parameter(texture_alpha, requires_grad=True)

        # Setup trainable RGB textures
        texture_size = texture_alpha.shape[-1]
        texture_color = torch.ones([npoints, self.num_channels, texture_size, texture_size]) * 0.1
        texture_color = texture_color.type(torch.float)
        texture_color = self.inverse_color_activation(texture_color.to("cuda"))
        self._texture_color = nn.Parameter(texture_color, requires_grad=True)

    def init_texture_color_from_spectral(self, data_dict, frame_id):
        faces = data_dict["faces"][frame_id].to(self.device)  # (F,3)
        verts = data_dict["points"][frame_id].to(self.device)  # (V,3)

        face_ch = spectral_face_channels(verts, faces, num_channels=self.num_channels, skip=1)
        bb_ch = face_ch[self._billboard_to_face]

        texture_size = self._texture_alpha.shape[-1]
        C = self.num_channels
        npoints = bb_ch.shape[0]

        tex = bb_ch[:, :, None, None].expand(npoints, C, texture_size, texture_size).contiguous()
        tex = tex.type(torch.float)

        tex = self.inverse_color_activation(tex)
        self._texture_color = nn.Parameter(tex, requires_grad=True)

    def _inference_mesh(self, data_dict, test_mode=False):
        Ps = torch.bmm(data_dict["camera"]["intrinsic"], data_dict["camera"]['extrinsic'][:, :-1])
        Ps = Ps.view(-1, 1, 3, 4)

        batch_size = len(data_dict['rgb'])
        head_pose_params = data_dict['model_params']['head']['pose_params']
        if test_mode:
            head_pose_params = head_pose_params.clone()

        # HEAD
        head_vs, landmarks3d, head_keys_proj = \
            FLAME_util.flame_forwarding(
                flame_model=self._flame_model,
                head_shape_params=data_dict['model_params']['head']['shape_params'],
                head_expression_params=data_dict['model_params']['head']['expression_params'],
                head_pose_params=head_pose_params,
                head_rotation=data_dict['model_params']['head']['global_orient'],
                head_transl=data_dict['model_params']['head']['transl'],
                head_scale_params=torch.ones((batch_size, 1)).to(self.device),
                device=self.device,
                img_size=(self._render_size, self._render_size),
                Ps=Ps,
                return2d=True,
            )

        # HAND
        rh_vs, rh_keys_3ds, rh_keys_proj = tracking_util.mano_forwarding(
            h_model=self._rh_model,
            betas=data_dict['model_params']['right_hand']['betas'],
            transl=data_dict['model_params']['right_hand']['transl'],
            rot=data_dict['model_params']['right_hand']['global_orient'],
            pose=data_dict['model_params']['right_hand']['hand_pose'],
            device=self.device,
            Ps=Ps,
            img_size=(self._render_size, self._render_size),
            return_2d=True,
        )

        RT = data_dict["camera"]['extrinsic']
        head_vs_in_cam = tranu.apply_transform(head_vs, RT=RT)
        rh_vs_in_cam = tranu.apply_transform(rh_vs, RT=RT)

        if self._use_deformations:
            deformations_cam = tranu.convert_canonical_deforms_into_cam_space(
                can_deforms=data_dict['deformation'],
                head_pose=data_dict['head_pose'],
                cam_transform=RT)
            deformed_head_vs_in_cam = head_vs_in_cam - deformations_cam
        else:
            deformed_head_vs_in_cam = head_vs_in_cam

        return {
            "head_vs": head_vs,
            "rh_vs_in_cam": rh_vs_in_cam,
            "deformed_head_vs_in_cam": deformed_head_vs_in_cam,
            "head_keys_proj": head_keys_proj[0],
            "rh_keys_proj": rh_keys_proj[0],
        }

    def _calc_faces_transform(self, data_dict, frame_id):
        faces = data_dict["faces"][frame_id]
        vertices = data_dict["points"][frame_id]

        T = torch.mean(vertices[faces], dim=1)

        sampled = vertices[faces]
        vec1 = sampled[:, 2] - sampled[:, 1]
        vec2 = sampled[:, 0] - sampled[:, 1]
        vec3 = sampled[:, 0] - sampled[:, 2]
        cross = torch.cross(vec1, vec2, dim=-1)
        norm = torch.nn.functional.normalize(cross, eps=1e-6, dim=-1)
        vec1 = torch.nn.functional.normalize(vec1, eps=1e-6, dim=-1)
        prod = torch.cross(vec1, norm, dim=-1)
        prod = torch.nn.functional.normalize(prod, eps=1e-6, dim=-1)
        rotmat = torch.stack([vec1, norm, prod], dim=-1)
        R = matrix_to_quaternion(rotmat)

        MAX_SCALE = 0.05
        area = torch.norm(cross, p=2, dim=-1, keepdim=True)
        vec3_length = torch.norm(vec3, p=2, dim=-1, keepdim=True)
        h = area / vec3_length
        k = torch.mean(torch.stack([h, vec3_length]), dim=0) / MAX_SCALE
        return T, R, k, norm

    def _calc_billboards_position(self, data_dict):
        batch_size = len(data_dict["faces"])

        xyz_list = []
        rotatation_list = []
        scaling_list = []
        for frame_id in range(batch_size):
            T, R, k, norm = self._calc_faces_transform(data_dict, frame_id)

            fidx = self._billboard_to_face                          # (N,)
            R_b = R[fidx]                                           # (N, 4)
            k_b = k[fidx]                                           # (N, 1)

            # Barycentric position + normal displacement (SplattingAvatar drifting attachment)
            faces_cur = data_dict["faces"][frame_id]                # (F, 3)
            verts_cur = data_dict["points"][frame_id]               # (V, 3)
            u = self._bary[:, 0] + self._xyz[:, 0]                 # u + Δu
            v = self._bary[:, 1] + self._xyz[:, 1]                 # v + Δv
            w = 1.0 - u - v
            tri_verts = verts_cur[faces_cur[fidx].long()].detach()    # (N, 3, 3)
            P = (u[:, None] * tri_verts[:, 0]
                 + v[:, None] * tri_verts[:, 1]
                 + w[:, None] * tri_verts[:, 2])                    # (N, 3)
            face_n = norm[fidx]                                     # (N, 3)
            xyz_list.append(P + self._xyz[:, 2:3] * face_n * k_b)

            rotation = self.rotation_activation(self._rotation)
            rigid_rotation = quaternion_multiply(R_b, rotation)
            rotatation_list.append(rigid_rotation)
            scaling_list.append(self.scaling_activation(self._scaling, self._min_s_value, self._max_s_value) * k_b)

        # Build texture_color (always 6 channels for rasterizer compatibility)
        base_color = self.color_activation(self._texture_color)  # (N, C, ts, ts)
        texture_color = base_color.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)

        predicted_parameters = {}
        predicted_parameters["xyz_offsets"] = self._xyz
        predicted_parameters["billboards_xyz"] = torch.stack(xyz_list, dim=0)
        predicted_parameters["billboards_rotations"] = torch.stack(rotatation_list, dim=0)
        predicted_parameters["billboards_scales"] = torch.stack(scaling_list, dim=0)
        predicted_parameters["billboards_colors"] = self._color.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        predicted_parameters["texture_alpha"] = self.opacity_activation(self._texture_alpha).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        predicted_parameters["texture_color"] = texture_color
        return  predicted_parameters

    def _precompute_face_adjacency(self):
        """Build per-face edge adjacency for FLAME and MANO meshes.
        adj[f, e] = global face index adjacent across edge e of face f, or -1 at boundaries.
        Edge convention: e=0↔V1-V2 (triggered when u<0),
                         e=1↔V0-V2 (v<0), e=2↔V0-V1 (w<0, i.e. u+v>1)."""
        n_flame = self._flame_faces.shape[0]
        n_rh = self._rh_faces.shape[0]
        adj = torch.full((n_flame + n_rh, 3), -1, dtype=torch.long)
        self._build_adjacency_for_faces(adj, self._flame_faces.long(), face_offset=0, n_faces=n_flame)
        self._build_adjacency_for_faces(adj, self._rh_faces.long(), face_offset=n_flame, n_faces=n_rh)
        self._face_adj = adj.to(self.device)

    def _build_adjacency_for_faces(self, adj, faces, face_offset, n_faces):
        edge_verts = [(1, 2), (0, 2), (0, 1)]
        faces_np = faces.cpu().numpy()
        edge_map = {}
        for i in range(n_faces):
            for e, (a, b) in enumerate(edge_verts):
                v0, v1 = int(faces_np[i, a]), int(faces_np[i, b])
                key = (min(v0, v1), max(v0, v1))
                if key in edge_map:
                    j, ej = edge_map.pop(key)
                    adj[i + face_offset, e] = j + face_offset
                    adj[j + face_offset, ej] = i + face_offset
                else:
                    edge_map[key] = (i, e)

    def walking_on_triangles(self, vertices, faces):
        """Walk Gaussians to neighboring faces when barycentric coords drift outside [0,1].
        Called every _triangle_walk_interval steps; absorbs (Δu,Δv) into _bary and resets _xyz[:,:2]."""
        vertices_np = vertices.cpu().float().numpy()
        faces_np = faces.cpu().numpy()
        adj_np = self._face_adj.cpu().numpy()

        face_idx = self._billboard_to_face.cpu().numpy().astype(np.int32)
        bary_np = self._bary.cpu().numpy()
        xyz_np = self._xyz.detach().cpu().numpy()

        u = bary_np[:, 0] + xyz_np[:, 0]
        v = bary_np[:, 1] + xyz_np[:, 1]
        w = 1.0 - u - v

        for i in np.where((u < 0) | (v < 0) | (w < 0))[0]:
            fi = int(face_idx[i])
            u_i, v_i, w_i = float(u[i]), float(v[i]), float(w[i])

            edge = int(np.argmin([u_i, v_i, w_i]))
            adj_fi = int(adj_np[fi, edge])

            if adj_fi < 0:
                u[i] = max(0.0, u_i)
                v[i] = max(0.0, min(v_i, 1.0 - u[i]))
                w[i] = 1.0 - u[i] - v[i]
                continue

            V = vertices_np[faces_np[fi]]
            P3d = u_i * V[0] + v_i * V[1] + w_i * V[2]

            V_adj = vertices_np[faces_np[adj_fi]]
            u[i], v[i], w[i] = _point_to_bary(P3d, V_adj[0], V_adj[1], V_adj[2])
            face_idx[i] = adj_fi

        self._billboard_to_face.data.copy_(torch.from_numpy(face_idx).long().to(self.device))
        self._bary.data[:, 0].copy_(torch.from_numpy(u.astype(np.float32)).to(self.device))
        self._bary.data[:, 1].copy_(torch.from_numpy(v.astype(np.float32)).to(self.device))
        self._bary.data[:, 2].copy_(torch.from_numpy(w.astype(np.float32)).to(self.device))
        self._xyz.data[:, :2].zero_()

    def _reset_optimizer_uv(self, xyz_optimizer):
        """Reset Adam momentum for _xyz[:, :2] after triangle walking."""
        for group in xyz_optimizer.param_groups:
            if group["name"] == "xyz":
                p = group["params"][0]
                if p in xyz_optimizer.state:
                    xyz_optimizer.state[p]["exp_avg"][:, :2].zero_()
                    xyz_optimizer.state[p]["exp_avg_sq"][:, :2].zero_()

    def make_optimizers(self, total_opt_step):
        xyz_optimizer = optim.Adam([{'params': [self._xyz], 'lr': 0.00016, 'name': 'xyz'}])
        scheduler = ExponentialLR(xyz_optimizer, gamma=0.01 ** (1.0 / total_opt_step))

        optimizer = optim.Adam([
            {'params': list(self._renderer.parameters()), 'lr': 0.001, 'name': 'renderer'},
            {'params': [self._color], 'lr': 0.0025, 'name': 'color'},
            {'params': [self._rotation], 'lr': 0.001, 'name': 'rotation'},
            {'params': [self._scaling], 'lr': 0.005, 'name': 'scaling'},
            {'params': [self._texture_alpha], 'lr': 0, 'name': 'texture_alpha'},
            {'params': [self._texture_color], 'lr': 0, 'name': 'texture_color'},
        ])

        return optimizer, xyz_optimizer, scheduler

    def activate_billboards_training(self, optimizer):
        if self._billboards_is_active:
            return
        for group in optimizer.param_groups:
            if group["name"] == "texture_alpha":
                group["lr"] = self._texture_opacity_lr
            if group["name"] == "texture_color":
                group["lr"] = self._texture_color_lr
        self._billboards_is_active = True
        print("Texture is activated")

    def _build_pose_cond(self, data_dict: dict) -> torch.Tensor:
        head  = data_dict['model_params']['head']
        rhand = data_dict['model_params']['right_hand']
        parts = [
            head['global_orient'].flatten(1),
            head['transl'].flatten(1),
            head['expression_params'].flatten(1),
            head['pose_params'].flatten(1),
            rhand['global_orient'].flatten(1),
            rhand['transl'].flatten(1),
            rhand['hand_pose'].flatten(1),
        ]
        return torch.cat(parts, dim=1)

    def render(self, data_dict, test_mode=False, flip=False):
        data_dict.update(self._inference_mesh(data_dict, test_mode))
        batch_size = len(data_dict["rgb"])

        points_head = data_dict["deformed_head_vs_in_cam"]
        points_hand = data_dict["rh_vs_in_cam"]
        points = torch.cat((points_head, points_hand), 1)
        faces = torch.cat((self._flame_faces, self._rh_faces + points_head.shape[1]), 0)
        faces = faces.unsqueeze(0).repeat(batch_size, 1, 1)

        data_dict.update({
            "points": points,
            "faces": faces,
        })

        # Rasterize billboards
        data_dict.update(self._calc_billboards_position(data_dict))
        data_dict["background"] = torch.permute(data_dict["background"], [0, 3, 1, 2])

        torch.cuda.synchronize()
        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)
        iter_start.record()

        data_dict.update(self._billboards_rasterizer(data_dict, roll_offset=(not test_mode)))
        features = data_dict["billboards_rasterization"]
        if self._use_renderer:
            features = features * data_dict["billboards_alpha"]

        # Optional: surface normals from rasterizer allmap[2:5] (view-space)
        if self._use_normals:
            features = torch.cat([features, data_dict["billboards_normal"]], dim=1)

        pose_cond = self._build_pose_cond(data_dict)
        if flip:
            features = torch.flip(features, dims=[-1])
        if self._use_renderer:
            color, alpha = self._renderer(features, pose_cond)
        else:
            # The caller is expected to have set data_dict["background"] to the
            # exact compositing color it wants, so the rasterizer output is
            # already the final image -- no further blending needed here.
            color, alpha = features, data_dict["billboards_alpha"]

        iter_end.record()
        torch.cuda.synchronize()
        run_time = iter_start.elapsed_time(iter_end)

        color = torch.permute(color, [0, 2, 3, 1])
        alpha = torch.permute(alpha, [0, 2, 3, 1])
        billboards_alpha = torch.permute(data_dict["billboards_alpha"], [0, 2, 3, 1])

        out = {
            "render_feautures": torch.permute(features, [0, 2, 3, 1]),
            "render_rgb": color,
            "render_mask": alpha,
            "billboards_alpha": billboards_alpha,
            "run_time": run_time,
        }


        return out

    def _run_mse_finetune(self, train_dataloader, ckpt_folder, global_step):
        print(f"MSE fine-tune: {self._mse_finetune_steps} steps, "
              f"lr={self._mse_finetune_lr}, soup_alpha={self._mse_soup_alpha}")

        original_renderer_state = {k: v.clone() for k, v in self._renderer.state_dict().items()}

        for param in self.parameters():
            param.requires_grad = False

        renderer_mse = copy.deepcopy(self._renderer)
        for param in renderer_mse.parameters():
            param.requires_grad = True
        self._renderer = renderer_mse

        ft_optimizer = optim.Adam(renderer_mse.parameters(), lr=self._mse_finetune_lr)
        ft_iter = iter(train_dataloader)

        self.train()
        pbar = tqdm(total=self._mse_finetune_steps, desc="MSE fine-tune")
        for step in range(self._mse_finetune_steps):
            try:
                batch = next(ft_iter)
            except StopIteration:
                ft_iter = iter(train_dataloader)
                batch = next(ft_iter)

            batch = dict2device(batch, self.device)
            batch["global_step"] = step
            b, h, w, c = batch["rgb"].shape
            bg = torch.rand(b, 1, 1, c, device=self.device).expand(b, h, w, c)
            batch["background"] = torch.full((b, h, w, self.num_channels), 0.5, device=self.device)

            batch.update(self.render(batch))

            render = batch["render_rgb"] * batch["render_mask"] + bg * (1 - batch["render_mask"])
            gt = batch["rgb"] * batch["mask"] + bg * (1 - batch["mask"])
            loss = ((gt - render) ** 2).mean()

            ft_optimizer.zero_grad()
            loss.backward()
            ft_optimizer.step()
            pbar.update(1)
            pbar.set_postfix({"mse": f"{loss.item():.5f}"})

        pbar.close()

        # Save pure MSE-finetuned checkpoint
        mse_ft_path = os.path.join(ckpt_folder, f"trainer_iter_{global_step}_mse_ft.pth")
        torch.save({"model_state_dict": self.state_dict()}, mse_ft_path)
        print(f"MSE fine-tuned checkpoint saved to {mse_ft_path}")

        # Blend: (1 - alpha) * original + alpha * mse-finetuned
        full_state = self.state_dict()
        soup_state = {}
        for k, v in full_state.items():
            if k.startswith("_renderer."):
                suffix = k[len("_renderer."):]
                orig_v = original_renderer_state.get(suffix, v)
                if v.is_floating_point() and v.shape == orig_v.shape:
                    soup_state[k] = (1 - self._mse_soup_alpha) * orig_v + self._mse_soup_alpha * v
                else:
                    soup_state[k] = orig_v
            else:
                soup_state[k] = v

        soup_path = os.path.join(ckpt_folder, f"trainer_iter_{global_step}_mse_soup.pth")
        torch.save({"model_state_dict": soup_state}, soup_path)
        print(f"MSE soup checkpoint saved to {soup_path}")

    def fit(self, train_dataset):
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size,
                                      shuffle=True, num_workers=4,
                                      persistent_workers=True, pin_memory=True)
        batch = next(iter(train_dataloader))
        batch = dict2device(batch, self.device)
        batch.update(self._inference_mesh(batch))
        points_head = batch["deformed_head_vs_in_cam"]
        points_hand = batch["rh_vs_in_cam"]
        points = torch.cat((points_head, points_hand), 1)
        faces = torch.cat((self._flame_faces, self._rh_faces + points_head.shape[1]), 0)
        faces = faces.unsqueeze(0)
        batch.update({
            "points": points,
            "faces": faces,
        })
        #self.init_texture_color_from_spectral(batch, 0)

        optimizer, xyz_optimizer, scheduler = self.make_optimizers(self.num_iterations)

        image_folder = os.path.join(self.log_folder, "renders")
        ckpt_folder = os.path.join(self.log_folder, "checkpoints")
        os.makedirs(ckpt_folder, exist_ok=True)
        os.makedirs(image_folder, exist_ok=True)

        global_step = 0
        data_iter = iter(train_dataloader)
        pbar = tqdm(total=self.num_iterations)
        while global_step < self.num_iterations:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_dataloader)
                batch = next(data_iter)

            pbar.update(1)
            batch = dict2device(batch, self.device)
            batch["global_step"] = global_step

            b, h, w, c = batch["rgb"].shape
            random_color = torch.rand(b, 1, 1, c).to(self.device)
            background_color = random_color.expand(b, h, w, c)

            if self._use_renderer:
                batch["background"] = torch.full((b, h, w, self.num_channels), 0.5).to(self.device)
                batch.update(self.render(batch))
                render = batch["render_rgb"] * batch["render_mask"] + background_color * (1 - batch["render_mask"])
            else:
                # No renderer to compensate for an arbitrary internal background,
                # so rasterize straight onto the color we want in the final image.
                batch["background"] = background_color
                batch.update(self.render(batch))
                render = batch["render_rgb"]

            gt = batch["rgb"] * batch["mask"] + background_color * (1 - batch["mask"])

            batch["rgb"] = gt
            batch["render_rgb"] = render

            batch["texture_alpha_init"] = self._texture_alpha_init
            batch["unet_l2"] = sum(p.pow(2).sum() for p in self._renderer.parameters())

            loss = 0
            log_loss = {}
            for criterion_name, criterion in self._criteria.items():
                local_loss = criterion(batch)
                log_loss[criterion.name] = local_loss.item()
                loss += local_loss

            pbar.set_postfix(log_loss)

            optimizer.zero_grad()
            xyz_optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            xyz_optimizer.step()
            scheduler.step()

            if self._triangle_walk and global_step > 0 and global_step % self._triangle_walk_interval == 0:
                with torch.no_grad():
                    self.walking_on_triangles(
                        batch["points"][0].detach(),
                        batch["faces"][0].detach()
                    )
                self._reset_optimizer_uv(xyz_optimizer)

            if global_step == 500:
                self.activate_billboards_training(optimizer)

            if global_step % self.log_iteration == 0:
                checkpoint_path = os.path.join(ckpt_folder, f"trainer_iter_{global_step}.pth")
                torch.save({
                    'iteration': global_step,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")

                log_image = render.cpu().detach().numpy()[0].copy()
                cv2.imwrite(os.path.join(image_folder, f"render_{global_step}.png"), log_image[..., ::-1] * 255)

                rgb_image = gt.cpu().detach().numpy()[0]
                cv2.imwrite(os.path.join(image_folder, f"gt_{global_step}.png"), rgb_image[..., ::-1] * 255)

                mask_image = batch["mask"].cpu().detach().numpy()[0]
                cv2.imwrite(os.path.join(image_folder, f"gt_mask_{global_step}.png"), mask_image * 255)

                mask_image = batch["billboards_alpha"].cpu().detach().numpy()[0]
                cv2.imwrite(os.path.join(image_folder, f"mask_{global_step}.png"), mask_image * 255)

                mask_image = batch["render_feautures"].cpu().detach().numpy()[0, ..., :3]
                cv2.imwrite(os.path.join(image_folder, f"feautures_{global_step}.png"), mask_image * 255)

            global_step += 1

        pbar.close()
        checkpoint_path = os.path.join(ckpt_folder, f"trainer_iter_{global_step}.pth")
        torch.save({
            'iteration': global_step,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

        if self._mse_finetune_steps > 0:
            self._run_mse_finetune(train_dataloader, ckpt_folder, global_step)
