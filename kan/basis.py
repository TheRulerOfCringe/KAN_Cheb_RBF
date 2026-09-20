# basis.py
import torch

def chebyshev_basis(x, x_min, x_max, degree):
    """
    x: (batch, in_dim)
    x_min, x_max: (in_dim,)
    degree: int
    return: (batch, in_dim, degree+1)
    """
    x_range = (x_max - x_min).clamp(min=1e-8)
    x_norm = 2.0 * (x - x_min) / x_range - 1.0
    x_norm = torch.clamp(x_norm, -1.0, 1.0)

    T = [torch.ones_like(x_norm)]
    if degree >= 1:
        T.append(x_norm)
    for n in range(2, degree + 1):
        T.append(2.0 * x_norm * T[-1] - T[-2])
    return torch.stack(T, dim=-1)

def rbf_basis(x, centers, sigma):
    """
    x: (batch, in_dim)
    centers: (in_dim, n_centers)
    sigma: scalar | (in_dim,) | (in_dim, n_centers)
    return: (batch, in_dim, n_centers)
    """
    diff = x[:, :, None] - centers[None, :, :]   # (batch, in_dim, n_centers)
    if isinstance(sigma, torch.Tensor):
        if sigma.dim() == 1:
            sigma = sigma[None, :, None]
        elif sigma.dim() == 2:
            sigma = sigma[None, :, :]
    return torch.exp(-0.5 * (diff / sigma) ** 2)

def fit_lstsq(B, y):
    """
    B: (batch, in_dim, n_basis)
    y: (batch, in_dim, out_dim)
    return: coef (in_dim, out_dim, n_basis)
    """
    in_dim = B.shape[1]
    out_dim = y.shape[-1]
    n_basis = B.shape[-1]
    device = B.device
    coef = torch.zeros(in_dim, out_dim, n_basis, device=device)
    B_perm = B.permute(1, 0, 2)   # (in_dim, batch, n_basis)
    y_perm = y.permute(1, 0, 2)   # (in_dim, batch, out_dim)
    for i in range(in_dim):
        sol = torch.linalg.lstsq(B_perm[i], y_perm[i]).solution  # (n_basis, out_dim)
        coef[i] = sol.T                                          # (out_dim, n_basis)
    return coef