import importlib

import cv2
import numpy as np
import torch


def get_obj_from_str(string, reload=False):
    """
    Return the class selected by a string

    :param string: String with path to the *.py file and class name
    :param reload: Flag to reload object
    :return: Selected class
    """
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config, additional_params=None):
    """
    Wrapper for object instantiation by it name in the config

    :param config: OmegaConf object with
    :return: Instantiated object with provided parameters
    """
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")

    config_params = config.get("params", dict())
    if additional_params:
        all_params = config_params.copy()
        all_params.update(additional_params)
    else:
        all_params = config_params

    return get_obj_from_str(config["target"])(**all_params)

  
def keypoint_overlay(kyes_2d,img,c=(0,255,0)): 
  
  [cv2.circle(img,(joint[0],joint[1]), 5, c, -1) for joint in kyes_2d]
  return img

def params_parser(params):
        #print(params.keys())  
        fids = list(params.keys())
        fids.sort() 
        
        parsed_dict = {"head":{},"right_hand":{}}
        for i,fid in enumerate(fids):
          
          if i==0:
            parsed_dict['head']['transl'] = torch.FloatTensor(params[fid]['head']['transl']).unsqueeze(0)
            parsed_dict['head']['global_orient'] = torch.FloatTensor(params[fid]['head']['global_orient']).unsqueeze(0)
            parsed_dict['head']['shape_params'] = torch.FloatTensor(params[fid]['head']['shape_params']).unsqueeze(0)
            parsed_dict['head']['pose_params'] = torch.FloatTensor(params[fid]['head']['pose_params']).unsqueeze(0)
            parsed_dict['head']['expression_params'] = torch.FloatTensor(params[fid]['head']['expression_params']).unsqueeze(0)
            parsed_dict['right_hand']['betas'] = torch.FloatTensor(params[fid]['right_hand']['betas']).unsqueeze(0)
            parsed_dict['right_hand']['global_orient'] = torch.FloatTensor(params[fid]['right_hand']['global_orient']).unsqueeze(0)
            parsed_dict['right_hand']['hand_pose'] = torch.FloatTensor(params[fid]['right_hand']['hand_pose']).unsqueeze(0)
            parsed_dict['right_hand']['transl'] = torch.FloatTensor(params[fid]['right_hand']['transl']).unsqueeze(0)
          else:
            parsed_dict['head']['transl'] = torch.cat((parsed_dict['head']['transl'],torch.FloatTensor(params[fid]['head']['transl']).unsqueeze(0)),dim=0)
            parsed_dict['head']['global_orient'] = torch.cat((parsed_dict['head']['global_orient'],torch.FloatTensor(params[fid]['head']['global_orient']).unsqueeze(0)),dim=0)
            parsed_dict['head']['shape_params'] = torch.cat((parsed_dict['head']['shape_params'],torch.FloatTensor(params[fid]['head']['shape_params']).unsqueeze(0)),dim=0)
            parsed_dict['head']['pose_params'] = torch.cat((parsed_dict['head']['pose_params'],torch.FloatTensor(params[fid]['head']['pose_params']).unsqueeze(0)),dim=0)
            parsed_dict['head']['expression_params'] = torch.cat((parsed_dict['head']['expression_params'],torch.FloatTensor(params[fid]['head']['expression_params']).unsqueeze(0)),dim=0)
            parsed_dict['right_hand']['betas'] = torch.cat((parsed_dict['right_hand']['betas'],torch.FloatTensor(params[fid]['right_hand']['betas']).unsqueeze(0)),dim=0)
            parsed_dict['right_hand']['global_orient'] = torch.cat((parsed_dict['right_hand']['global_orient'],torch.FloatTensor(params[fid]['right_hand']['global_orient']).unsqueeze(0)),dim=0)
            parsed_dict['right_hand']['hand_pose'] = torch.cat((parsed_dict['right_hand']['hand_pose'],torch.FloatTensor(params[fid]['right_hand']['hand_pose']).unsqueeze(0)),dim=0)
            parsed_dict['right_hand']['transl'] = torch.cat((parsed_dict['right_hand']['transl'],torch.FloatTensor(params[fid]['right_hand']['transl']).unsqueeze(0)),dim=0)
            
        return parsed_dict 
      
      
def denormalize_keys_batch(keys,w,h):
    """
     keys:Bxn_viewsxNx2
    """
    keys[:,:,:,0]*=w
    keys[:,:,:,1]*=h
    return keys

C0 = 0.28209479177387814

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def q_normalize(q):
    """
    Normalize the coefficients of a given quaternion tensor of shape [*, 4].
    """
    assert q.shape[-1] == 4

    norm = torch.sqrt(torch.sum(torch.square(q), dim=-1))  # ||q|| = sqrt(w²+x²+y²+z²)
    assert not torch.any(
        torch.isclose(norm, torch.zeros_like(norm, device=q.device)))  # check for singularities
    return torch.div(q, norm[:, None])  # q_norm = q / ||q||s

def s_act(x, min_s_value, max_s_value):
    if isinstance(x, float):
        x = torch.tensor(x).squeeze()
    return min_s_value + torch.sigmoid(x) * (max_s_value - min_s_value)


def s_inv_act(x, min_s_value, max_s_value):
    if isinstance(x, float):
        x = torch.tensor(x).squeeze()
    y = (x - min_s_value) / (max_s_value - min_s_value) + 1e-5
    y = torch.logit(y)
    assert not torch.isnan(
        y
    ).any(), f"{x.min()}, {x.max()}, {y.min()}, {y.max()}"
    return y

def inverse_sigmoid(x):
    return torch.log(x/(1-x) + 1e-6)

def depths_to_points(viewmatrix, tanfovx, tanfovy, render_size, depthmap):
    c2w = (viewmatrix.T).inverse()
    W, H = render_size[0], render_size[1]
    fx = W / (2 * tanfovx)
    fy = H / (2 * tanfovy)
    intrins = torch.tensor(
        [[fx, 0., W/2.],
        [0., fy, H/2.],
        [0., 0., 1.0]]
    ).float().cuda()
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
    rays_o = c2w[:3,3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points

def depth_to_normal(viewmatrix, tanfovx, tanfovy, render_size, depth):
    """
        view: view camera
        depth: depthmap
    """
    points = depths_to_points(viewmatrix, tanfovx, tanfovy, render_size, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output
