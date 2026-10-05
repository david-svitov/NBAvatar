import torch


def cotangent(a, b, eps=1e-12):
    cross = torch.linalg.norm(torch.cross(a, b, dim=-1), dim=-1).clamp_min(eps)
    dot = (a * b).sum(dim=-1)
    return dot / cross

def build_cotan_laplacian(vertices, faces):
    faces = faces.to(torch.long)
    V = vertices.shape[0]
    device, dtype = vertices.device, vertices.dtype

    i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2]
    v0, v1, v2 = vertices[i0], vertices[i1], vertices[i2]

    def cotangent(a, b, eps=1e-12):
        cross = torch.linalg.norm(torch.cross(a, b, dim=-1), dim=-1).clamp_min(eps)
        dot = (a * b).sum(dim=-1)
        return dot / cross

    cot0 = cotangent(v1 - v0, v2 - v0)
    cot1 = cotangent(v2 - v1, v0 - v1)
    cot2 = cotangent(v0 - v2, v1 - v2)

    w01 = 0.5 * cot2
    w12 = 0.5 * cot0
    w20 = 0.5 * cot1

    L = torch.zeros((V, V), device=device, dtype=dtype)

    def add_edge_acc(ii, jj, w):
        # off-diagonal (-w)
        L.index_put_((ii, jj), -w, accumulate=True)
        L.index_put_((jj, ii), -w, accumulate=True)
        # diagonal (+w)
        L.index_put_((ii, ii),  w, accumulate=True)
        L.index_put_((jj, jj),  w, accumulate=True)

    add_edge_acc(i0, i1, w01)
    add_edge_acc(i1, i2, w12)
    add_edge_acc(i2, i0, w20)

    # mass
    face_area = 0.5 * torch.linalg.norm(torch.cross(v1 - v0, v2 - v0, dim=-1), dim=-1)
    M = torch.zeros((V,), device=device, dtype=dtype)
    area_third = face_area / 3.0
    M.scatter_add_(0, i0, area_third)
    M.scatter_add_(0, i1, area_third)
    M.scatter_add_(0, i2, area_third)

    return L, M

@torch.no_grad()
def spectral_face_channels(vertices, faces, num_channels=6, skip=1):
    vertices = vertices.contiguous()
    faces = faces.to(dtype=torch.long).contiguous()

    L, M = build_cotan_laplacian(vertices, faces)
    eps = 1e-12

    Minv_sqrt = (M.clamp_min(eps)).rsqrt()
    A = (Minv_sqrt[:, None] * L) * Minv_sqrt[None, :]

    evals, evecs = torch.linalg.eigh(A)

    Y = evecs[:, skip:skip + num_channels]
    X = Minv_sqrt[:, None] * Y

    phi_f = X[faces].mean(dim=1)

    mn = phi_f.min(dim=0).values
    mx = phi_f.max(dim=0).values
    out = (phi_f - mn) / (mx - mn).clamp_min(1e-12)
    return out.clamp(0.0, 1.0)
