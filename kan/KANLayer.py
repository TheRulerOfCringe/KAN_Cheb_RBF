# KANLayer.py
import torch
import torch.nn as nn
import numpy as np
from .spline import extend_grid, coef2curve, curve2coef
from .utils import sparse_mask
from .basis import chebyshev_basis, rbf_basis, fit_lstsq


class KANLayer(nn.Module):
    """
    Базис выбирается параметром basis:
      'bspline' : обычные B-сплайны (как раньше)
      'cheb'    : полиномы Чебышева; grid = (in_dim, 2) = [x_min, x_max];
                  coef = (in_dim, out_dim, degree+1)
      'rbf'     : гауссовы RBF; grid = (in_dim, n_centers);
                  coef = (in_dim, out_dim, n_centers); sigma — буфер
    """
    def __init__(self, in_dim=3, out_dim=2, num=5, k=3, noise_scale=0.5,
                 scale_base_mu=0.0, scale_base_sigma=1.0, scale_sp=1.0,
                 base_fun=torch.nn.SiLU(), grid_eps=0.02, grid_range=[-1, 1],
                 sp_trainable=True, sb_trainable=True, save_plot_data=True,
                 device='cpu', sparse_init=False,
                 basis='bspline', degree=None, sigma=None):
        super(KANLayer, self).__init__()
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.num = num
        self.k = k
        self.basis = basis
        self.grid_eps = grid_eps

        if basis == 'bspline':
            grid = torch.linspace(grid_range[0], grid_range[1], steps=num + 1)[None, :].expand(self.in_dim, num + 1)
            grid = extend_grid(grid, k_extend=k)
            self.grid = torch.nn.Parameter(grid).requires_grad_(False)
            noises = (torch.rand(self.num + 1, self.in_dim, self.out_dim) - 0.5) * noise_scale / num
            self.coef = torch.nn.Parameter(curve2coef(self.grid[:, k:-k].permute(1, 0), noises, self.grid, k))

        elif basis == 'cheb':
            self.degree = int(degree) if degree is not None else int(num)
            grid = torch.zeros(in_dim, 2)
            grid[:, 0] = grid_range[0]
            grid[:, 1] = grid_range[1]
            self.grid = torch.nn.Parameter(grid).requires_grad_(False)
            self.coef = torch.nn.Parameter(
                (torch.rand(in_dim, out_dim, self.degree + 1) - 0.5) * noise_scale
            )

        elif basis == 'rbf':
            self.num_centers = int(num)
            centers = torch.linspace(grid_range[0], grid_range[1], steps=self.num_centers)[None, :] \
                        .expand(in_dim, self.num_centers).clone()
            self.grid = torch.nn.Parameter(centers).requires_grad_(False)
            if sigma is None:
                if self.num_centers > 1:
                    sigma = (grid_range[1] - grid_range[0]) / (self.num_centers - 1)
                else:
                    sigma = 1.0
            self.register_buffer('sigma', torch.tensor(float(sigma)))
            self.coef = torch.nn.Parameter(
                (torch.rand(in_dim, out_dim, self.num_centers) - 0.5) * noise_scale
            )
        else:
            raise ValueError(f"Unknown basis: {basis}")

        if sparse_init:
            self.mask = torch.nn.Parameter(sparse_mask(in_dim, out_dim)).requires_grad_(False)
        else:
            self.mask = torch.nn.Parameter(torch.ones(in_dim, out_dim)).requires_grad_(False)

        self.scale_base = torch.nn.Parameter(
            scale_base_mu * 1 / np.sqrt(in_dim)
            + scale_base_sigma * (torch.rand(in_dim, out_dim) * 2 - 1) * 1 / np.sqrt(in_dim)
        ).requires_grad_(sb_trainable)
        self.scale_sp = torch.nn.Parameter(
            torch.ones(in_dim, out_dim) * scale_sp * 1 / np.sqrt(in_dim) * self.mask
        ).requires_grad_(sp_trainable)
        self.base_fun = base_fun
        self.to(device)

    def to(self, device):
        super(KANLayer, self).to(device)
        self.device = device
        return self

    # --- ядро базиса --------------------------------------------------------

    def _eval_output(self, x):
        """Возвращает (batch, in_dim, out_dim) — «сырой» выход базиса (без scale_base/scale_sp/mask)."""
        if self.basis == 'bspline':
            return coef2curve(x_eval=x, grid=self.grid, coef=self.coef, k=self.k)
        elif self.basis == 'cheb':
            T = chebyshev_basis(x, self.grid[:, 0], self.grid[:, 1], self.degree)
            return torch.einsum('bic,ioc->bio', T, self.coef)
        elif self.basis == 'rbf':
            B = rbf_basis(x, self.grid, self.sigma)
            return torch.einsum('bic,ioc->bio', B, self.coef)

    def _fit(self, x, y):
        """Подгонка coef под y(x). Возвращает (in_dim, out_dim, n_basis)."""
        if self.basis == 'bspline':
            return curve2coef(x, y, self.grid, self.k)
        elif self.basis == 'cheb':
            T = chebyshev_basis(x, self.grid[:, 0], self.grid[:, 1], self.degree)
            return fit_lstsq(T, y)
        elif self.basis == 'rbf':
            B = rbf_basis(x, self.grid, self.sigma)
            return fit_lstsq(B, y)

    # --- forward ------------------------------------------------------------

    def forward(self, x):
        batch = x.shape[0]
        preacts = x[:, None, :].clone().expand(batch, self.out_dim, self.in_dim)
        base = self.base_fun(x)

        y = self._eval_output(x)
        postspline = y.clone().permute(0, 2, 1)

        y = self.scale_base[None, :, :] * base[:, :, None] + self.scale_sp[None, :, :] * y
        y = self.mask[None, :, :] * y

        postacts = y.clone().permute(0, 2, 1)
        y = torch.sum(y, dim=1)
        return y, preacts, postacts, postspline

    # --- grid updates -------------------------------------------------------

    def update_grid_from_samples(self, x, mode='sample'):
        batch = x.shape[0]
        x_pos = torch.sort(x, dim=0)[0]
        y_eval = self._eval_output(x_pos)   # фиксируем функцию ДО обновления

        if self.basis == 'bspline':
            num_interval = self.grid.shape[1] - 1 - 2 * self.k

            def get_grid(num_interval):
                ids = [int(batch / num_interval * i) for i in range(num_interval)] + [-1]
                grid_adaptive = x_pos[ids, :].permute(1, 0)
                margin = 0.00
                h = (grid_adaptive[:, [-1]] - grid_adaptive[:, [0]] + 2 * margin) / num_interval
                grid_uniform = grid_adaptive[:, [0]] - margin + h * torch.arange(
                    num_interval + 1, device=x.device)[None, :]
                return self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive

            grid = get_grid(num_interval)
            if mode == 'grid':
                sample_grid = get_grid(2 * num_interval)
                x_pos = sample_grid.permute(1, 0)
                y_eval = coef2curve(x_pos, self.grid, self.coef, self.k)
            self.grid.data = extend_grid(grid, k_extend=self.k)
            self.coef.data = curve2coef(x_pos, y_eval, self.grid, self.k)

        elif self.basis == 'cheb':
            # адаптируем домен к данным, затем переобучаем коэффициенты
            x_min, x_max = x_pos[0, :], x_pos[-1, :]
            old_min, old_max = self.grid.data[:, 0], self.grid.data[:, 1]
            new_min = self.grid_eps * old_min + (1 - self.grid_eps) * x_min
            new_max = self.grid_eps * old_max + (1 - self.grid_eps) * x_max
            self.grid.data = torch.stack([new_min, new_max], dim=1)
            self.coef.data = self._fit(x_pos, y_eval)

        elif self.basis == 'rbf':
            # переставляем центры по квантилям и обновляем sigma
            n_centers = self.num_centers
            ids = [int(batch / n_centers * i) for i in range(n_centers)]
            new_centers = x_pos[ids, :].permute(1, 0)
            self.grid.data = self.grid_eps * self.grid.data + (1 - self.grid_eps) * new_centers
            if n_centers > 1:
                diffs = self.grid.data[:, 1:] - self.grid.data[:, :-1]
                self.sigma.data = torch.abs(diffs).mean()
            self.coef.data = self._fit(x_pos, y_eval)

    def initialize_grid_from_parent(self, parent, x, mode='sample'):
        batch = x.shape[0]
        x_pos = torch.sort(x, dim=0)[0]
        y_eval_parent = parent._eval_output(x_pos)

        if self.basis != parent.basis:
            raise ValueError("parent and child must use the same basis")

        if self.basis == 'bspline':
            # оригинальная логика интерполяции сетки родителя
            def get_grid(num_interval):
                x_pos_p = parent.grid[:, parent.k:-parent.k]
                sp2 = KANLayer(in_dim=1, out_dim=self.in_dim, k=1,
                               num=x_pos_p.shape[1] - 1,
                               scale_base_mu=0.0, scale_base_sigma=0.0).to(x.device)
                sp2_coef = curve2coef(
                    sp2.grid[:, sp2.k:-sp2.k].permute(1, 0).expand(-1, self.in_dim),
                    x_pos_p.permute(1, 0).unsqueeze(dim=2),
                    sp2.grid[:, :], k=1
                ).permute(1, 0, 2)
                sp2.coef.data = sp2_coef
                percentile = torch.linspace(-1, 1, self.num + 1).to(self.device)
                return sp2(percentile.unsqueeze(dim=1))[0].permute(1, 0)

            num_interval = self.grid.shape[1] - 1 - 2 * self.k
            grid = get_grid(num_interval)
            if mode == 'grid':
                sample_grid = get_grid(2 * num_interval)
                x_pos = sample_grid.permute(1, 0)
                y_eval = coef2curve(x_pos, parent.grid, parent.coef, parent.k)
            else:
                y_eval = y_eval_parent
            self.grid.data = extend_grid(grid, k_extend=self.k)
            self.coef.data = curve2coef(x_pos, y_eval, self.grid, self.k)

        elif self.basis == 'cheb':
            # домен наследуем у родителя; degree может отличаться
            self.grid.data = parent.grid.data.clone()
            self.coef.data = self._fit(x_pos, y_eval_parent)

        elif self.basis == 'rbf':
            if self.num_centers != parent.num_centers:
                # интерполируем центры родителя на нужное число ядер
                new_centers = torch.zeros(self.in_dim, self.num_centers, device=x.device)
                for i in range(self.in_dim):
                    new_centers[i] = torch.nn.functional.interpolate(
                        parent.grid.data[i].view(1, 1, -1),
                        size=self.num_centers, mode='linear', align_corners=True
                    ).view(-1)
                self.grid.data = new_centers
                if self.num_centers > 1:
                    diffs = new_centers[:, 1:] - new_centers[:, :-1]
                    self.sigma.data = torch.abs(diffs).mean()
            else:
                self.grid.data = parent.grid.data.clone()
                self.sigma.data = parent.sigma.data.clone()
            self.coef.data = self._fit(x_pos, y_eval_parent)

    # --- утилиты ------------------------------------------------------------

    def get_subset(self, in_id, out_id):
        spb = KANLayer(
            len(in_id), len(out_id), self.num, self.k, base_fun=self.base_fun,
            basis=self.basis,
            degree=getattr(self, 'degree', None),
            sigma=float(self.sigma) if self.basis == 'rbf' else None,
        )
        spb.grid.data = self.grid[in_id]
        spb.coef.data = self.coef[in_id][:, out_id]
        spb.scale_base.data = self.scale_base[in_id][:, out_id]
        spb.scale_sp.data = self.scale_sp[in_id][:, out_id]
        spb.mask.data = self.mask[in_id][:, out_id]
        spb.in_dim = len(in_id)
        spb.out_dim = len(out_id)
        return spb

    def swap(self, i1, i2, mode='in'):
        with torch.no_grad():
            def swap_(data, i1, i2, mode='in'):
                if mode == 'in':
                    data[i1], data[i2] = data[i2].clone(), data[i1].clone()
                elif mode == 'out':
                    data[:, i1], data[:, i2] = data[:, i2].clone(), data[:, i1].clone()

            if mode == 'in':
                swap_(self.grid.data, i1, i2, mode='in')
            swap_(self.coef.data, i1, i2, mode=mode)
            swap_(self.scale_base.data, i1, i2, mode=mode)
            swap_(self.scale_sp.data, i1, i2, mode=mode)
            swap_(self.mask.data, i1, i2, mode=mode)