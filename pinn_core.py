import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import trange

_SEED_SET = False

def set_seed(seed: int = 42, force: bool = False):
    global _SEED_SET
    if _SEED_SET and not force:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    _SEED_SET = True

class SinActivation(nn.Module):

    def forward(self, x):
        return torch.sin(x)

class InputNormalizer:

    def __init__(self, domain: dict):
        self.domain = domain
        self.var_names = list(domain.keys())

    def normalize_tensor(self, x: torch.Tensor) -> torch.Tensor:
        normed_cols = []
        for i, name in enumerate(self.var_names):
            low, high = self.domain[name]
            span = high - low
            col = x[:, i:i + 1]
            if span < 1e-12:
                normed_cols.append(torch.zeros_like(col))
            else:
                normed_cols.append(2.0 * (col - low) / span - 1.0)
        return torch.cat(normed_cols, dim=1)

class PINN(nn.Module):

    def __init__(self, input_dim=1, output_dim=1, hidden_size=64,
                 num_hidden_layers=4, activation='tanh',
                 use_residual=False):
        super().__init__()
        self.use_residual = use_residual

        self.input_layer = nn.Linear(input_dim, hidden_size)
        self.input_activation = self._make_activation(activation)
        self.hidden_layers = nn.ModuleList()
        self.hidden_activations = nn.ModuleList()
        for _ in range(num_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
            self.hidden_activations.append(self._make_activation(activation))
        self.output_layer = nn.Linear(hidden_size, output_dim)
        self._init_weights()

    @staticmethod
    def _make_activation(name):
        return {'tanh': nn.Tanh, 'sin': SinActivation,
                'swish': nn.SiLU, 'gelu': nn.GELU}[name]()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.input_activation(self.input_layer(x))
        for layer, act in zip(self.hidden_layers, self.hidden_activations):
            h = act(layer(h)) + h if self.use_residual else act(layer(h))
        return self.output_layer(h)

def grad(y, x, order=1):
    result = y
    for _ in range(order):
        result = torch.autograd.grad(
            result, x, grad_outputs=torch.ones_like(result),
            create_graph=True, retain_graph=True)[0]
    return result

class LossWeighter:

    def __init__(self, method='fixed', lambda_res=1.0,
                 lambda_cond=10.0, adapt_rate=0.1):
        self.method = method
        self.lambda_res = lambda_res
        self.lambda_cond = lambda_cond
        self.adapt_rate = adapt_rate
        self._prev_res = None
        self._prev_cond = None

    def get_weights(self, loss_res, loss_cond, model=None):
        if self.method == 'fixed':
            return self.lambda_res, self.lambda_cond
        elif self.method == 'softadapt':
            return self._softadapt(loss_res, loss_cond)
        elif self.method == 'grad_norm':
            return self._grad_norm(loss_res, loss_cond, model)
        return self.lambda_res, self.lambda_cond

    def _softadapt(self, loss_res, loss_cond):
        res_val, cond_val = loss_res.item(), loss_cond.item()
        if self._prev_res is not None:
            rate_res = res_val - self._prev_res
            rate_cond = cond_val - self._prev_cond
            mx = max(rate_res, rate_cond)
            er = np.exp(rate_res - mx)
            ec = np.exp(rate_cond - mx)
            s = er + ec + 1e-12
            a = self.adapt_rate
            self.lambda_res = (1 - a) * self.lambda_res + a * (2 * er / s)
            self.lambda_cond = (1 - a) * self.lambda_cond + a * (2 * ec / s)
        self._prev_res = res_val
        self._prev_cond = cond_val
        return self.lambda_res, self.lambda_cond

    def _grad_norm(self, loss_res, loss_cond, model):
        if model is None:
            return self.lambda_res, self.lambda_cond

        params = [p for p in model.parameters() if p.requires_grad]

        g_res = torch.autograd.grad(loss_res, params,
                                    retain_graph=True,
                                    allow_unused=True)
        norm_res = sum(p.norm() for p in g_res if p is not None)
        norm_res = torch.clamp(norm_res, min=1e-6)

        g_cond = torch.autograd.grad(loss_cond, params,
                                     retain_graph=True,
                                     allow_unused=True)
        norm_cond = sum(p.norm() for p in g_cond if p is not None)
        norm_cond = torch.clamp(norm_cond, min=1e-6)

        mean_norm = (norm_res + norm_cond) / 2.0
        desired_res = (mean_norm / norm_res).item()
        desired_cond = (mean_norm / norm_cond).item()

        desired_res = min(max(desired_res, 0.01), 100.0)
        desired_cond = min(max(desired_cond, 0.01), 100.0)

        a = self.adapt_rate
        self.lambda_res = (1 - a) * self.lambda_res + a * desired_res
        self.lambda_cond = (1 - a) * self.lambda_cond + a * desired_cond
        return self.lambda_res, self.lambda_cond

class PINNSolver:

    def __init__(self, equation, domain, conditions,
                 input_dim=None, output_dim=1,
                 hidden_size=64, num_hidden_layers=4,
                 activation='tanh',
                 use_residual=False,
                 loss_weighting='fixed',
                 lambda_res=1.0, lambda_cond=10.0,
                 domain_mask=None,
                 seed=42, device=None):
        set_seed(seed)
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.equation = equation
        self.domain = domain
        self.conditions = self._normalize_conditions(conditions)
        self.domain_mask = domain_mask
        self.var_names = list(domain.keys())

        if input_dim is None:
            input_dim = len(self.var_names)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.normalizer = InputNormalizer(domain)
        self.model = PINN(
            input_dim=input_dim, output_dim=output_dim,
            hidden_size=hidden_size, num_hidden_layers=num_hidden_layers,
            activation=activation,
            use_residual=use_residual
        ).to(self.device)

        self.weighter = LossWeighter(
            method=loss_weighting,
            lambda_res=lambda_res, lambda_cond=lambda_cond)

        self.history = {'total': [], 'residual': [], 'conditions': [],
                        'lambda_res': [], 'lambda_cond': [],
                        'validation': []}

        self._hparams = dict(
            input_dim=input_dim, output_dim=output_dim,
            hidden_size=hidden_size, num_hidden_layers=num_hidden_layers,
            activation=activation,
            use_residual=use_residual, loss_weighting=loss_weighting,
            lambda_res=lambda_res, lambda_cond=lambda_cond)

    @staticmethod
    def _normalize_conditions(conditions):
        out = []
        for c in conditions:
            c = dict(c)
            if 'type' not in c:
                if 'deriv_var' in c:
                    c['type'] = 'neumann'
                elif 'var' in c and 'where' not in c:
                    c['type'] = 'periodic'
                else:
                    c['type'] = 'dirichlet'
            out.append(c)
        return out

    def _make_collocation_points(self, n_coll, sampling='random'):
        ndim = len(self.var_names)

        if ndim >= 3 and sampling == 'grid':
            sampling = 'random'

        coords_raw = {}
        if sampling == 'random':
            for v in self.var_names:
                lo, hi = self.domain[v]
                t = (lo + (hi - lo) * torch.rand(
                    n_coll, 1, device=self.device)
                ).detach().requires_grad_(True)
                coords_raw[v] = t
        elif sampling == 'grid':
            n_per = max(int(n_coll ** (1.0 / ndim)), 10)
            if ndim == 1:
                v = self.var_names[0]
                lo, hi = self.domain[v]
                coords_raw[v] = torch.linspace(
                    lo, hi, n_per, device=self.device
                ).reshape(-1, 1).detach().requires_grad_(True)
            else:
                grids = []
                for v in self.var_names:
                    lo, hi = self.domain[v]
                    grids.append(torch.linspace(
                        lo, hi, n_per, device=self.device))
                mesh = torch.meshgrid(*grids, indexing='ij')
                for i, v in enumerate(self.var_names):
                    coords_raw[v] = mesh[i].flatten().reshape(
                        -1, 1).detach().requires_grad_(True)

        if self.domain_mask is not None:
            coords_raw = self._apply_mask(coords_raw)

        raw = torch.cat([coords_raw[v] for v in self.var_names], dim=1)
        normed = self.normalizer.normalize_tensor(raw)
        return normed, coords_raw

    def _apply_mask(self, coords_raw):
        mask = self.domain_mask(coords_raw)
        if hasattr(mask, 'flatten'):
            mask = mask.flatten()
        if mask.all():
            return coords_raw
        out = {}
        for v in self.var_names:
            flat = coords_raw[v].flatten()
            vals = flat[mask].detach().reshape(-1, 1).requires_grad_(True)
            out[v] = vals
        return out

    def _make_condition_tensors(self, cond, n_pts):
        if 'custom_coords' in cond:
            cc = cond['custom_coords']
            coords_raw = {}
            for v in self.var_names:
                arr = np.asarray(cc[v]).ravel().reshape(-1, 1).astype(np.float32)
                t = torch.tensor(arr, device=self.device).requires_grad_(True)
                coords_raw[v] = t
            raw = torch.cat([coords_raw[v] for v in self.var_names], dim=1)
            normed = self.normalizer.normalize_tensor(raw)
            return normed, coords_raw

        where = cond.get('where', {})
        free = [v for v in self.var_names if v not in where]
        coords_raw = {}

        if len(free) == 0:
            for v in self.var_names:
                coords_raw[v] = torch.tensor(
                    [[where[v]]], dtype=torch.float32,
                    device=self.device).requires_grad_(True)
        elif len(free) == 1:
            for v in self.var_names:
                if v in where:
                    coords_raw[v] = torch.full(
                        (n_pts, 1), where[v], dtype=torch.float32,
                        device=self.device).requires_grad_(True)
                else:
                    lo, hi = self.domain[v]
                    coords_raw[v] = torch.linspace(
                        lo, hi, n_pts, device=self.device
                    ).reshape(-1, 1).detach().requires_grad_(True)
        else:
            n_per = max(int(n_pts ** (1.0 / len(free))), 5)
            free_grids = []
            for v in free:
                lo, hi = self.domain[v]
                free_grids.append(torch.linspace(
                    lo, hi, n_per, device=self.device))
            mesh = torch.meshgrid(*free_grids, indexing='ij')
            fi = 0
            for v in self.var_names:
                if v in where:
                    n_total = mesh[0].numel()
                    coords_raw[v] = torch.full(
                        (n_total, 1), where[v], dtype=torch.float32,
                        device=self.device).requires_grad_(True)
                else:
                    coords_raw[v] = mesh[fi].flatten().reshape(
                        -1, 1).detach().requires_grad_(True)
                    fi += 1

        raw = torch.cat([coords_raw[v] for v in self.var_names], dim=1)
        normed = self.normalizer.normalize_tensor(raw)
        return normed, coords_raw

    def _make_periodic_tensors(self, cond, n_pts):
        pvar = cond['var']
        lo, hi = self.domain[pvar]
        free = [v for v in self.var_names if v != pvar]

        if len(free) == 0:

            coords_lo = {pvar: torch.tensor(
                [[lo]], dtype=torch.float32,
                device=self.device).requires_grad_(True)}
            coords_hi = {pvar: torch.tensor(
                [[hi]], dtype=torch.float32,
                device=self.device).requires_grad_(True)}
        else:
            n_per = max(int(n_pts ** (1.0 / len(free))), 5)
            free_grids = []
            for v in free:
                flo, fhi = self.domain[v]
                free_grids.append(torch.linspace(
                    flo, fhi, n_per, device=self.device))
            if len(free) == 1:
                free_flat = [free_grids[0].reshape(-1, 1)]
            else:
                mesh = torch.meshgrid(*free_grids, indexing='ij')
                free_flat = [m.flatten().reshape(-1, 1) for m in mesh]
            n_total = free_flat[0].shape[0]

            coords_lo = {}
            coords_hi = {}
            fi = 0
            for v in self.var_names:
                if v == pvar:
                    coords_lo[v] = torch.full(
                        (n_total, 1), lo, dtype=torch.float32,
                        device=self.device).requires_grad_(True)
                    coords_hi[v] = torch.full(
                        (n_total, 1), hi, dtype=torch.float32,
                        device=self.device).requires_grad_(True)
                else:
                    coords_lo[v] = free_flat[fi].detach().to(
                        self.device).requires_grad_(True)
                    coords_hi[v] = free_flat[fi].detach().to(
                        self.device).requires_grad_(True)
                    fi += 1

        raw_lo = torch.cat([coords_lo[v] for v in self.var_names], dim=1)
        raw_hi = torch.cat([coords_hi[v] for v in self.var_names], dim=1)
        norm_lo = self.normalizer.normalize_tensor(raw_lo)
        norm_hi = self.normalizer.normalize_tensor(raw_hi)
        return norm_lo, coords_lo, norm_hi, coords_hi

    def _compute_loss(self, n_coll, n_cond, sampling='random'):

        normed, coords = self._make_collocation_points(n_coll, sampling)
        u = self.model(normed)
        D = lambda u_c, var, order=1: grad(u_c, var, order)
        residual = self.equation(u, coords, D)
        if isinstance(residual, (list, tuple)):
            loss_res = sum(torch.mean(r ** 2) for r in residual)
        else:
            loss_res = torch.mean(residual ** 2)

        loss_cond = torch.tensor(0.0, device=self.device)

        for c in self.conditions:
            ctype = c['type']
            comp = c.get('component', 0)

            if ctype == 'periodic':
                loss_cond = loss_cond + self._loss_periodic(c, n_cond, comp)
            elif ctype == 'robin':
                loss_cond = loss_cond + self._loss_robin(c, n_cond, comp)
            elif ctype == 'neumann':
                loss_cond = loss_cond + self._loss_neumann(c, n_cond, comp)
            else:
                loss_cond = loss_cond + self._loss_dirichlet(c, n_cond, comp)

        wr, wc = self.weighter.get_weights(
            loss_res, loss_cond, model=self.model)
        total = wr * loss_res + wc * loss_cond
        return total, loss_res, loss_cond, wr, wc

    def _loss_dirichlet(self, c, n_cond, comp):
        ci, cc = self._make_condition_tensors(c, n_cond)
        uc = self.model(ci)
        u_comp = uc[:, comp:comp + 1]
        target = self._resolve_target(c['value'], cc, u_comp)
        return torch.mean((u_comp - target) ** 2)

    def _loss_neumann(self, c, n_cond, comp):
        ci, cc = self._make_condition_tensors(c, n_cond)
        uc = self.model(ci)
        u_comp = uc[:, comp:comp + 1]
        du = grad(u_comp, cc[c['deriv_var']])
        target = self._resolve_target(c['value'], cc, du)
        return torch.mean((du - target) ** 2)

    def _loss_robin(self, c, n_cond, comp):
        ci, cc = self._make_condition_tensors(c, n_cond)
        uc = self.model(ci)
        u_comp = uc[:, comp:comp + 1]
        du = grad(u_comp, cc[c['deriv_var']])
        alpha = c.get('alpha', 1.0)
        beta = c.get('beta', 1.0)
        lhs = alpha * u_comp + beta * du
        target = self._resolve_target(c['value'], cc, lhs)
        return torch.mean((lhs - target) ** 2)

    def _loss_periodic(self, c, n_cond, comp):
        n_lo, c_lo, n_hi, c_hi = self._make_periodic_tensors(c, n_cond)
        u_lo = self.model(n_lo)[:, comp:comp + 1]
        u_hi = self.model(n_hi)[:, comp:comp + 1]
        loss = torch.mean((u_lo - u_hi) ** 2)

        if c.get('match_deriv', False):
            pvar = c['var']
            du_lo = grad(u_lo, c_lo[pvar])
            du_hi = grad(u_hi, c_hi[pvar])
            loss = loss + torch.mean((du_lo - du_hi) ** 2)
        return loss

    @staticmethod
    def _resolve_target(value, coords, like_tensor):
        if callable(value):
            t = value(coords)
            if isinstance(t, np.ndarray):
                t = torch.tensor(t, dtype=torch.float32,
                                 device=like_tensor.device).reshape(-1, 1)
            return t
        return torch.full_like(like_tensor, value)

    def solve(self, n_epochs_adam=5000, n_collocation=200,
              n_condition=50, lr=1e-3, use_lbfgs=True,
              lbfgs_max_iter=3000, sampling='random',
              scheduler_type='cosine', early_stop_patience=2000,
              early_stop_rtol=1e-9,
              validation_fn=None, validation_interval=500,
              verbose=True):
        self.history = {'total': [], 'residual': [], 'conditions': [],
                        'lambda_res': [], 'lambda_cond': [],
                        'validation': []}
        t0 = time.time()

        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        sched = None
        if scheduler_type == 'cosine':
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=n_epochs_adam, eta_min=1e-6)

        if verbose:
            print(f'Adam ({n_epochs_adam} ep, lr={lr}, '
                  f'coll={n_collocation} {sampling})')

        for ep in trange(n_epochs_adam, desc='Adam', disable=not verbose):
            opt.zero_grad()
            tot, res, cnd, wr, wc = self._compute_loss(
                n_collocation, n_condition, sampling)
            tot.backward()
            opt.step()
            if sched:
                sched.step()

            self._record(tot, res, cnd, wr, wc)

            if verbose and (ep + 1) % 2000 == 0:
                print(f'  {ep + 1:5d} | L={tot.item():.2e} | '
                      f'res={res.item():.2e} | cond={cnd.item():.2e}')

            if validation_fn and (ep + 1) % validation_interval == 0:
                val = validation_fn(self)
                self.history['validation'].append(
                    {'epoch': ep + 1, 'value': val})
                if verbose:
                    print(f'  [val] epoch {ep + 1}: {val:.4e}')

        if use_lbfgs:
            if verbose:
                print(f'L-BFGS (max {lbfgs_max_iter})')
            ol = torch.optim.LBFGS(
                self.model.parameters(), lr=1.0,
                max_iter=20, max_eval=25,
                tolerance_grad=1e-9, tolerance_change=1e-12,
                history_size=50, line_search_fn='strong_wolfe')
            it = [0]
            best = [float('inf')]
            noimpr = [0]

            lbfgs_sampling = 'grid' if len(self.var_names) < 3 else 'random'

            def closure():
                ol.zero_grad()
                tot, res, cnd, wr, wc = self._compute_loss(
                    n_collocation, n_condition, lbfgs_sampling)
                tot.backward()
                it[0] += 1
                self._record(tot, res, cnd, wr, wc)

                val = tot.item()

                rel_impr = (best[0] - val) / (abs(best[0]) + 1e-12)
                if rel_impr > early_stop_rtol:
                    best[0] = val
                    noimpr[0] = 0
                else:
                    noimpr[0] += 1

                if verbose and it[0] % 200 == 0:
                    print(f'  L-BFGS {it[0]:5d} | L={val:.2e}')
                return tot

            for _ in range(lbfgs_max_iter // 20):
                ol.step(closure)
                if noimpr[0] >= early_stop_patience\
                        or it[0] >= lbfgs_max_iter:
                    break

        elapsed = time.time() - t0
        tot, res, cnd, _, _ = self._compute_loss(
            n_collocation, n_condition,
            'grid' if len(self.var_names) < 3 else 'random')
        if verbose:
            print(f'Done {elapsed:.1f}s | res={res.item():.2e} '
                  f'| cond={cnd.item():.2e} | total={tot.item():.2e}')
        return self.history

    def _record(self, tot, res, cnd, wr, wc):
        self.history['total'].append(tot.item())
        self.history['residual'].append(res.item())
        self.history['conditions'].append(cnd.item())
        self.history['lambda_res'].append(
            wr if isinstance(wr, (int, float)) else wr)
        self.history['lambda_cond'].append(
            wc if isinstance(wc, (int, float)) else wc)

    def predict(self, *coords_arrays):
        with torch.no_grad():
            ts = [torch.tensor(a, dtype=torch.float32,
                               device=self.device).reshape(-1, 1)
                  if isinstance(a, np.ndarray)
                  else a.to(self.device).reshape(-1, 1)
                  for a in coords_arrays]
            xr = torch.cat(ts, dim=1)
            xn = self.normalizer.normalize_tensor(xr)
            return self.model(xn).cpu().numpy()

    def predict_grid(self, n=100):
        grids = {}
        for v in self.var_names:
            lo, hi = self.domain[v]
            grids[v] = np.linspace(lo, hi, n)
        mesh = np.meshgrid(
            *[grids[v] for v in self.var_names], indexing='ij')
        flat = [m.flatten() for m in mesh]
        u_pred = self.predict(*flat)
        shape = tuple(n for _ in self.var_names)
        mesh_dict = {v: mesh[i] for i, v in enumerate(self.var_names)}
        return u_pred.reshape(*shape, -1), grids, mesh_dict

    def evaluate(self, exact_fn, n_test=500, metric='l2_rel'):
        tc = {}
        for v in self.var_names:
            lo, hi = self.domain[v]
            tc[v] = np.linspace(lo, hi, n_test)
        if len(self.var_names) == 1:
            u_pred = self.predict(tc[self.var_names[0]])
            u_exact = exact_fn(tc[self.var_names[0]])
        else:
            mesh = np.meshgrid(
                *[tc[v] for v in self.var_names], indexing='ij')
            flat = [m.flatten() for m in mesh]
            u_pred = self.predict(*flat)
            u_exact = exact_fn(*flat)
        if isinstance(u_exact, np.ndarray) and u_exact.ndim == 1:
            u_exact = u_exact.reshape(-1, 1)
        diff = u_pred - u_exact
        if metric == 'l2_rel':
            err = np.linalg.norm(diff) / max(np.linalg.norm(u_exact), 1e-12)
        else:
            err = np.max(np.abs(diff))
        print(f'  {metric}: {err:.4e}')
        return {'error': err, 'metric': metric,
                'u_pred': u_pred, 'u_exact': u_exact}

    def save(self, path: str):
        path = Path(path)
        data = {
            'model_state': self.model.state_dict(),
            'hparams': self._hparams,
            'domain': self.domain,
            'history': self.history,
            'var_names': self.var_names,
        }
        torch.save(data, path)
        print(f'Saved to {path}  ({path.stat().st_size / 1024:.0f} KB)')

    @classmethod
    def load(cls, path: str, equation, conditions,
             domain_mask=None, device=None):
        data = torch.load(path, map_location=device or 'cpu',
                          weights_only=False)
        hp = data['hparams']
        solver = cls(
            equation=equation,
            domain=data['domain'],
            conditions=conditions,
            domain_mask=domain_mask,
            device=device,
            **hp)
        solver.model.load_state_dict(data['model_state'])
        solver.history = data.get('history', solver.history)
        print(f'Loaded from {path}')
        return solver

print('PINN Solver loaded')
