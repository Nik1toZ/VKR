_EPS_ZERO = 1e-10


def _compute_errors(u_pred, u_exact):

    u_pred = np.asarray(u_pred).ravel()
    u_exact = np.asarray(u_exact).ravel()
    valid = np.isfinite(u_pred) & np.isfinite(u_exact)
    u_pred = u_pred[valid]
    u_exact = u_exact[valid]
    diff = u_pred - u_exact
    abs_max = float(np.max(np.abs(diff))) if diff.size else 0.0
    abs_l2 = float(np.linalg.norm(diff))
    norm_exact = float(np.linalg.norm(u_exact))
    rel_defined = norm_exact >= _EPS_ZERO
    rel_l2 = abs_l2 / norm_exact if rel_defined else abs_max
    return {
        'rel_l2': rel_l2, 'abs_max': abs_max, 'abs_l2': abs_l2,
        'norm_exact': norm_exact, 'rel_defined': rel_defined,
    }


def _rel_l2(u_pred, u_exact):
    
    return _compute_errors(u_pred, u_exact)['rel_l2']


def _err_summary(u_pred, u_exact, indent='  '):
    
    e = _compute_errors(u_pred, u_exact)
    if e['rel_defined']:
        return (f"{indent}Rel L2: {e['rel_l2']:.3e}   "
                f"Max |Δ|: {e['abs_max']:.3e}")
    return (f"{indent}Max |Δ|: {e['abs_max']:.3e}   "
            f"(Rel L2 не определена: аналит. ≈ 0)")


def plot_2d_surface_pair(X, Y, Upred, Uexact, title='',
                         xlabel='x', ylabel='t', zlabel='u'):
    fig = plt.figure(figsize=(16, 5))
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.plot_surface(X, Y, Upred, cmap='viridis', alpha=0.9,
                     rstride=2, cstride=2, edgecolor='none')
    ax1.set_title(f'{title} — PINN')
    ax1.set_xlabel(xlabel); ax1.set_ylabel(ylabel); ax1.set_zlabel(zlabel)
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.plot_surface(X, Y, Uexact, cmap='viridis', alpha=0.9,
                     rstride=2, cstride=2, edgecolor='none')
    ax2.set_title(f'{title} — аналит.')
    ax2.set_xlabel(xlabel); ax2.set_ylabel(ylabel); ax2.set_zlabel(zlabel)
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.plot_surface(X, Y, np.abs(Upred - Uexact), cmap='plasma', alpha=0.9,
                     rstride=2, cstride=2, edgecolor='none')
    ax3.set_title(f'{title} — |ошибка|')
    ax3.set_xlabel(xlabel); ax3.set_ylabel(ylabel); ax3.set_zlabel('|Δ|')
    plt.tight_layout(); plt.show(); plt.close('all')
    print(_err_summary(Upred, Uexact))


def plot_slices(x, slices_pred, slices_exact, slice_values, slice_name,
                xlabel='x', ylabel='u', title=''):
    n = len(slice_values)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n))
    for i, sv in enumerate(slice_values):
        axes[0].plot(x, slices_exact[i], color=colors[i], ls='-',  lw=2.0,
                     label=f'{slice_name}={sv:.2f} аналит.')
        axes[0].plot(x, slices_pred[i],  color=colors[i], ls='--', lw=1.6,
                     label=f'{slice_name}={sv:.2f} PINN')
    axes[0].set_xlabel(xlabel); axes[0].set_ylabel(ylabel)
    axes[0].set_title(f'{title} — срезы'); axes[0].legend(fontsize=7, ncol=2)
    for i, sv in enumerate(slice_values):
        err = np.abs(slices_pred[i].ravel() - slices_exact[i].ravel())
        axes[1].semilogy(x, np.maximum(err, 1e-16), color=colors[i], lw=1.8,
                         label=f'{slice_name}={sv:.2f}')
    axes[1].set_xlabel(xlabel); axes[1].set_ylabel('|Δ|')
    axes[1].set_title(f'{title} — ошибка (log)'); axes[1].legend(fontsize=8)
    plt.tight_layout(); plt.show(); plt.close('all')


def plot_loss_history(history, title=''):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.semilogy(history['total'],     label='total',    alpha=0.8)
    ax.semilogy(history['residual'],  label='residual', alpha=0.8)
    ax.semilogy(history['conditions'],label='BC/IC',    alpha=0.8)
    ax.set_xlabel('iteration'); ax.set_ylabel('loss')
    ax.set_title(f'{title} — история обучения'); ax.legend()
    plt.tight_layout(); plt.show(); plt.close('all')




def _enumerate_faces(domain):
    faces = []
    var_names = list(domain.keys())
    for v in var_names:
        lo, hi = domain[v]
        free = [u for u in var_names if u != v]
        faces.append((v, 'lo', lo, free))
        faces.append((v, 'hi', hi, free))
    return faces


def _face_coords_2d(domain, face_var, face_val, free, N):
    assert len(free) == 2
    u, v = free
    lo_u, hi_u = domain[u]; lo_v, hi_v = domain[v]
    su = np.linspace(lo_u, hi_u, N)
    sv = np.linspace(lo_v, hi_v, N)
    U, V = np.meshgrid(su, sv, indexing='ij')
    coords = {face_var: np.full_like(U, face_val), u: U, v: V}
    return U, V, u, v, coords


def _predict_from_coords(solver, coords):
    order = solver.var_names
    flats = [np.asarray(coords[v]).ravel() for v in order]
    up = solver.predict(*flats)
    shape = np.asarray(coords[order[0]]).shape
    return up.reshape(*shape, -1) if up.shape[-1] > 1 else up.reshape(*shape)


def plot_all_boundaries_3d(solver, exact_fn, domain, title='', N=40):
    
    faces = _enumerate_faces(domain)
    errs = {}
    for fv, side, val, free in faces:
        U, V, uvar, vvar, coords = _face_coords_2d(domain, fv, val, free, N)
        up = _predict_from_coords(solver, coords)
        ue = exact_fn(**coords)
        fig = plt.figure(figsize=(16, 4.2))
        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        ax1.plot_surface(U, V, up, cmap='viridis', alpha=0.9,
                         rstride=2, cstride=2, edgecolor='none')
        ax1.set_title(f'{fv}={val:.3g} ({side})  PINN')
        ax1.set_xlabel(uvar); ax1.set_ylabel(vvar)
        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        ax2.plot_surface(U, V, ue, cmap='viridis', alpha=0.9,
                         rstride=2, cstride=2, edgecolor='none')
        ax2.set_title(f'{fv}={val:.3g} ({side})  аналит.')
        ax2.set_xlabel(uvar); ax2.set_ylabel(vvar)
        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        ax3.plot_surface(U, V, np.abs(up - ue), cmap='plasma', alpha=0.9,
                         rstride=2, cstride=2, edgecolor='none')
        ax3.set_title(f'{fv}={val:.3g} ({side})  |ошибка|')
        ax3.set_xlabel(uvar); ax3.set_ylabel(vvar)
        plt.suptitle(f'{title} — грань {fv}={val:.3g}', y=1.02)
        plt.tight_layout(); plt.show(); plt.close('all')
        errs[f'{fv}={val:.3g} ({side})'] = _compute_errors(up, ue)
    return errs


def plot_all_boundaries_4d(solver, exact_fn, domain, title='', N=30):
    
    faces = _enumerate_faces(domain)
    errs = {}
    for fv, side, val, free in faces:
        free_sorted = list(free)
        fix_var = free_sorted[-1]
        fix_lo, fix_hi = domain[fix_var]
        fix_val = 0.5 * (fix_lo + fix_hi)
        u_var, v_var = free_sorted[0], free_sorted[1]
        lo_u, hi_u = domain[u_var]; lo_v, hi_v = domain[v_var]
        su = np.linspace(lo_u, hi_u, N)
        sv = np.linspace(lo_v, hi_v, N)
        U, V = np.meshgrid(su, sv, indexing='ij')
        coords = {
            fv: np.full_like(U, val),
            fix_var: np.full_like(U, fix_val),
            u_var: U, v_var: V,
        }
        up = _predict_from_coords(solver, coords)
        ue = exact_fn(**coords)
        fig = plt.figure(figsize=(16, 4.2))
        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        ax1.plot_surface(U, V, up, cmap='viridis', alpha=0.9,
                         rstride=2, cstride=2, edgecolor='none')
        ax1.set_title(f'{fv}={val:.3g}, {fix_var}={fix_val:.3g}  PINN')
        ax1.set_xlabel(u_var); ax1.set_ylabel(v_var)
        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        ax2.plot_surface(U, V, ue, cmap='viridis', alpha=0.9,
                         rstride=2, cstride=2, edgecolor='none')
        ax2.set_title(f'{fv}={val:.3g}  аналит.')
        ax2.set_xlabel(u_var); ax2.set_ylabel(v_var)
        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        ax3.plot_surface(U, V, np.abs(up - ue), cmap='plasma', alpha=0.9,
                         rstride=2, cstride=2, edgecolor='none')
        ax3.set_title(f'{fv}={val:.3g}  |ошибка|')
        ax3.set_xlabel(u_var); ax3.set_ylabel(v_var)
        plt.suptitle(
            f'{title} — грань {fv}={val:.3g} (срез по {fix_var}={fix_val:.3g})',
            y=1.02)
        plt.tight_layout(); plt.show(); plt.close('all')
        n_full = 20
        sw = np.linspace(*domain[fix_var], n_full)
        su2 = np.linspace(*domain[u_var], n_full)
        sv2 = np.linspace(*domain[v_var], n_full)
        U3, V3, W3 = np.meshgrid(su2, sv2, sw, indexing='ij')
        coords3 = {
            fv: np.full_like(U3, val),
            fix_var: W3, u_var: U3, v_var: V3,
        }
        up3 = _predict_from_coords(solver, coords3)
        ue3 = exact_fn(**coords3)
        errs[f'{fv}={val:.3g} ({side})'] = _compute_errors(up3, ue3)
    return errs



def plot_interior_scatter_3d(solver, exact_fn, domain, n_points=4000,
                             title='', mask_fn=None):
    rng = np.random.default_rng(0)
    var_names = solver.var_names
    ndim = len(var_names)

    oversample = 3 if mask_fn is not None else 1
    pts = {}
    for v in var_names:
        lo, hi = domain[v]
        span = hi - lo
        eps = 0.03 * span
        pts[v] = rng.uniform(lo + eps, hi - eps, n_points * oversample)

    if mask_fn is not None:
        m = mask_fn({v: pts[v] for v in var_names})
        for v in var_names:
            pts[v] = pts[v][m][:n_points]
    else:
        for v in var_names:
            pts[v] = pts[v][:n_points]

    up = _predict_from_coords(solver, pts).ravel()
    ue = exact_fn(**pts).ravel()
    err = np.abs(up - ue)

    if ndim == 2:
        fig, ax = plt.subplots(figsize=(6.5, 5))
        sc = ax.scatter(pts[var_names[0]], pts[var_names[1]], c=err,
                        cmap='plasma', s=8, alpha=0.8)
        ax.set_xlabel(var_names[0]); ax.set_ylabel(var_names[1])
        ax.set_title(f'{title}: |ошибка| во внутренних точках')
        ax.set_aspect('equal', adjustable='box')
        plt.colorbar(sc, ax=ax, label='|u - u*|')
        plt.tight_layout(); plt.show(); plt.close('all')
    elif ndim == 3:
        fig = plt.figure(figsize=(7.5, 6))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(pts[var_names[0]], pts[var_names[1]], pts[var_names[2]],
                        c=err, cmap='plasma', s=6, alpha=0.65)
        ax.set_xlabel(var_names[0]); ax.set_ylabel(var_names[1]); ax.set_zlabel(var_names[2])
        ax.set_title(f'{title}: |ошибка| в {",".join(var_names)}')
        plt.colorbar(sc, ax=ax, label='|u - u*|', shrink=0.7)
        plt.tight_layout(); plt.show(); plt.close('all')
    elif ndim == 4:
        v0, v1, v2, v3 = var_names
        fig = plt.figure(figsize=(15, 6))
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        sc1 = ax1.scatter(pts[v0], pts[v1], pts[v2],
                          c=err, cmap='plasma', s=6, alpha=0.6)
        ax1.set_xlabel(v0); ax1.set_ylabel(v1); ax1.set_zlabel(v2)
        ax1.set_title(f'ошибка в ({v0},{v1},{v2}), цвет зависит и от {v3}')
        plt.colorbar(sc1, ax=ax1, label='|Δ|', shrink=0.7)
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        sc2 = ax2.scatter(pts[v1], pts[v2], pts[v3],
                          c=err, cmap='plasma', s=6, alpha=0.6)
        ax2.set_xlabel(v1); ax2.set_ylabel(v2); ax2.set_zlabel(v3)
        ax2.set_title(f'ошибка в ({v1},{v2},{v3}), цвет зависит и от {v0}')
        plt.colorbar(sc2, ax=ax2, label='|Δ|', shrink=0.7)
        plt.suptitle(f'{title}: ошибка во внутренних точках', y=1.02)
        plt.tight_layout(); plt.show(); plt.close('all')
    print(_err_summary(up, ue))
    print(f'  Max |Δ| внутри: {err.max():.3e}   Median |Δ|: {np.median(err):.3e}')


def plot_center_slices(solver, exact_fn, domain, title='', N=200,
                       mask_fn=None):
    var_names = solver.var_names
    k = len(var_names)
    has_exact = exact_fn is not None
    ncols = 2 if has_exact else 1
    fig, axes = plt.subplots(k, ncols, figsize=(13 if has_exact else 7,
                                                 3.2 * k))
    if k == 1:
        axes = np.array(axes).reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    for i, v in enumerate(var_names):
        lo, hi = domain[v]
        s = np.linspace(lo, hi, N)
        coords = {v: s}
        for other in var_names:
            if other == v:
                continue
            olo, ohi = domain[other]
            coords[other] = np.full_like(s, 0.5 * (olo + ohi))
        up = _predict_from_coords(solver, coords).ravel()
        ue = exact_fn(**coords).ravel() if has_exact else None

        if mask_fn is not None:
            m = mask_fn(coords).ravel()
            up = np.where(m, up, np.nan)
            if ue is not None:
                ue = np.where(m, ue, np.nan)

        fixed_str = ', '.join(
            f'{u}={0.5*(domain[u][0]+domain[u][1]):.2f}'
            for u in var_names if u != v)

        if has_exact:
            axes[i,0].plot(s, ue, 'k-',  lw=2.2, label='аналит.')
            axes[i,0].plot(s, up, 'r--', lw=1.6, label='PINN')
            axes[i,0].set_xlabel(v); axes[i,0].set_ylabel('u')
            axes[i,0].set_title(f'u({v}) при {fixed_str}')
            axes[i,0].legend(fontsize=8)
            err = np.abs(up - ue)
            axes[i,1].semilogy(s, np.maximum(err, 1e-16), 'b-', lw=1.7)
            axes[i,1].set_xlabel(v); axes[i,1].set_ylabel('|u - u*|')
            axes[i,1].set_title(f'Ошибка u({v}) (log)')
        else:
            axes[i,0].plot(s, up, 'r-', lw=1.8, label='PINN')
            axes[i,0].set_xlabel(v); axes[i,0].set_ylabel('u')
            axes[i,0].set_title(f'u({v}) при {fixed_str}')
            axes[i,0].legend(fontsize=8)

    note = ''
    if mask_fn is not None:
        note = ' (точки вне области скрыты)'
    plt.suptitle(f'{title} — центральные срезы по каждой переменной{note}',
                 y=1.002)
    plt.tight_layout(); plt.show(); plt.close('all')


plot_interior_scatter = plot_interior_scatter_3d


def report_boundary_errors_table(errs_dict, title=''):
    
    print(f'\n=== {title}: ошибки на каждой грани ===')
    first = next(iter(errs_dict.values()), None)
    if isinstance(first, dict):
        print(f'  {"грань":35s}  {"rel L2":>12s}  {"max |Δ|":>12s}  примечание')
        print(f'  {"-"*35}  {"-"*12}  {"-"*12}  {"-"*30}')
        for k, e in errs_dict.items():
            if e['rel_defined']:
                note = ''
            else:
                note = 'rel не определена (аналит.≈0)'
            rel_str = f"{e['rel_l2']:.3e}" if e['rel_defined'] else '   —   '
            print(f"  {k:35s}  {rel_str:>12s}  {e['abs_max']:12.3e}  {note}")
    else:
        for k, v in errs_dict.items():
            print(f'  {k:35s}  {v:.3e}')
    print()




def plot_masked_2d(solver, domain, mask_fn, title='', N=120,
                   exact_fn=None, component=0):

    var_names = solver.var_names
    assert len(var_names) == 2
    vx, vy = var_names
    lox, hix = domain[vx]; loy, hiy = domain[vy]
    xs = np.linspace(lox, hix, N)
    ys = np.linspace(loy, hiy, N)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    coords = {vx: X, vy: Y}
    m = mask_fn(coords)
    up_raw = solver.predict(X.ravel(), Y.ravel())
    up = up_raw[:, component].reshape(X.shape) if up_raw.shape[1] > 1 \
        else up_raw.reshape(X.shape)
    up_masked = np.where(m, up, np.nan)

    n_panels = 3 if exact_fn is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 4.6))
    if n_panels == 1:
        axes = [axes]
    cf = axes[0].contourf(X, Y, up_masked, levels=40, cmap='viridis')
    axes[0].set_title(f'{title} — PINN')
    axes[0].set_xlabel(vx); axes[0].set_ylabel(vy)
    axes[0].set_aspect('equal', adjustable='box')
    plt.colorbar(cf, ax=axes[0])
    if exact_fn is not None:
        ue = exact_fn(**coords)
        ue_masked = np.where(m, ue, np.nan)
        cf2 = axes[1].contourf(X, Y, ue_masked, levels=40, cmap='viridis')
        axes[1].set_title(f'{title} — аналит.')
        axes[1].set_xlabel(vx); axes[1].set_ylabel(vy)
        axes[1].set_aspect('equal', adjustable='box')
        plt.colorbar(cf2, ax=axes[1])
        err_masked = np.where(m, np.abs(up - ue), np.nan)
        cf3 = axes[2].contourf(X, Y, err_masked, levels=40, cmap='plasma')
        axes[2].set_title(f'{title} — |ошибка|')
        axes[2].set_xlabel(vx); axes[2].set_ylabel(vy)
        axes[2].set_aspect('equal', adjustable='box')
        plt.colorbar(cf3, ax=axes[2])
        valid_mask = m & np.isfinite(ue)
        print(_err_summary(up[valid_mask], ue[valid_mask]))
    plt.tight_layout(); plt.show(); plt.close('all')


def plot_parametric_family(solver, exact_fn, domain, param_name,
                           param_values, title='', N=80):

    var_names = solver.var_names
    spatial = [v for v in var_names if v != param_name]
    assert len(spatial) == 2
    vx, vy = spatial
    lox, hix = domain[vx]; loy, hiy = domain[vy]
    xs = np.linspace(lox, hix, N)
    ys = np.linspace(loy, hiy, N)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    n = len(param_values)
    fig, axes = plt.subplots(3, n, figsize=(4.2 * n, 11),
                              subplot_kw={'projection': '3d'})
    if n == 1:
        axes = axes.reshape(3, 1)
    errs = []
    for i, pv in enumerate(param_values):
        coords = {vx: X, vy: Y, param_name: np.full_like(X, pv)}
        up = _predict_from_coords(solver, coords)
        ue = exact_fn(**coords)
        # Строка 1: PINN
        axes[0, i].plot_surface(X, Y, up, cmap='viridis', alpha=0.9,
                                rstride=2, cstride=2, edgecolor='none')
        axes[0, i].set_title(f'PINN @ {param_name}={pv:.2g}')
        axes[0, i].set_xlabel(vx); axes[0, i].set_ylabel(vy)
        # Строка 2: аналит.
        axes[1, i].plot_surface(X, Y, ue, cmap='viridis', alpha=0.9,
                                rstride=2, cstride=2, edgecolor='none')
        axes[1, i].set_title(f'аналит. @ {param_name}={pv:.2g}')
        axes[1, i].set_xlabel(vx); axes[1, i].set_ylabel(vy)
        # Строка 3: |Δ|
        axes[2, i].plot_surface(X, Y, np.abs(up - ue), cmap='plasma', alpha=0.9,
                                rstride=2, cstride=2, edgecolor='none')
        axes[2, i].set_title(f'|Δ| @ {param_name}={pv:.2g}')
        axes[2, i].set_xlabel(vx); axes[2, i].set_ylabel(vy)
        errs.append(_compute_errors(up, ue))
    plt.suptitle(f'{title} — семейство по {param_name}', y=1.002)
    plt.tight_layout(); plt.show(); plt.close('all')
    print(f'  Ошибки на разных {param_name}:')
    for pv, e in zip(param_values, errs):
        if e['rel_defined']:
            print(f"    {param_name}={pv:.3g}: "
                  f"rel L2 = {e['rel_l2']:.3e}   max |Δ| = {e['abs_max']:.3e}")
        else:
            print(f"    {param_name}={pv:.3g}: "
                  f"max |Δ| = {e['abs_max']:.3e}   (rel не определена)")


def plot_time_snapshots(solver, exact_fn, domain, t_values,
                        title='', N=300):
    
    var_names = solver.var_names
    spatial = [v for v in var_names if v != 't']
    assert len(spatial) == 1
    vx = spatial[0]
    lox, hix = domain[vx]
    xs = np.linspace(lox, hix, N)
    slices_pred = []
    slices_exact = []
    for t_val in t_values:
        coords = {vx: xs, 't': np.full_like(xs, t_val)}
        up = _predict_from_coords(solver, coords).ravel()
        ue = exact_fn(**coords).ravel() if exact_fn is not None else None
        slices_pred.append(up)
        slices_exact.append(ue if ue is not None else np.full_like(up, np.nan))
    if exact_fn is not None:
        plot_slices(xs, slices_pred, slices_exact, t_values, 't',
                    xlabel=vx, title=title)
    else:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(t_values)))
        for i, t_val in enumerate(t_values):
            ax.plot(xs, slices_pred[i], color=colors[i], lw=1.8,
                    label=f't={t_val:.2f}')
        ax.set_xlabel(vx); ax.set_ylabel('u')
        ax.set_title(f'{title} — временные снимки')
        ax.legend(fontsize=9)
        plt.tight_layout(); plt.show(); plt.close('all')


def plot_vector_field_2d(solver, domain, mask_fn=None, title='',
                         N=30, comp_u=0, comp_v=1, comp_p=2):
    
    var_names = solver.var_names
    vx, vy = var_names[0], var_names[1]
    lox, hix = domain[vx]; loy, hiy = domain[vy]
    xs = np.linspace(lox, hix, N * 3)
    ys = np.linspace(loy, hiy, N)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    pred = solver.predict(X.ravel(), Y.ravel())
    u_val = pred[:, comp_u].reshape(X.shape)
    v_val = pred[:, comp_v].reshape(X.shape)
    p_val = pred[:, comp_p].reshape(X.shape)
    mag = np.sqrt(u_val ** 2 + v_val ** 2)
    if mask_fn is not None:
        m = mask_fn({vx: X, vy: Y})
        u_val = np.where(m, u_val, np.nan)
        v_val = np.where(m, v_val, np.nan)
        p_val = np.where(m, p_val, np.nan)
        mag = np.where(m, mag, np.nan)
    fig, axes = plt.subplots(3, 1, figsize=(13, 10))
    cf0 = axes[0].contourf(X, Y, mag, levels=30, cmap='viridis')
    step = max(1, N // 15)
    axes[0].quiver(X[::step, ::step], Y[::step, ::step],
                   np.where(np.isnan(u_val), 0, u_val)[::step, ::step],
                   np.where(np.isnan(v_val), 0, v_val)[::step, ::step],
                   color='white', alpha=0.85, scale=25)
    axes[0].set_title(f'{title} — |скорость| и направление')
    axes[0].set_xlabel(vx); axes[0].set_ylabel(vy)
    axes[0].set_aspect('equal', adjustable='box')
    plt.colorbar(cf0, ax=axes[0])
    cf1 = axes[1].contourf(X, Y, u_val, levels=30, cmap='RdBu_r')
    axes[1].set_title(f'{title} — u (горизонтальная скорость)')
    axes[1].set_xlabel(vx); axes[1].set_ylabel(vy)
    axes[1].set_aspect('equal', adjustable='box')
    plt.colorbar(cf1, ax=axes[1])
    cf2 = axes[2].contourf(X, Y, p_val, levels=30, cmap='coolwarm')
    axes[2].set_title(f'{title} — давление p')
    axes[2].set_xlabel(vx); axes[2].set_ylabel(vy)
    axes[2].set_aspect('equal', adjustable='box')
    plt.colorbar(cf2, ax=axes[2])
    plt.tight_layout(); plt.show(); plt.close('all')




def plot_cross_sections_ns(solver, domain, x_positions,
                           mask_fn=None, title='',
                           comp_u=0, comp_v=1, N=120,
                           reference_profiles=None):
    
    var_names = solver.var_names
    vx, vy = var_names[0], var_names[1]
    loy, hiy = domain[vy]
    ys = np.linspace(loy, hiy, N)

    n_pos = len(x_positions)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_pos))

    for i, x_val in enumerate(x_positions):
        x_col = np.full_like(ys, x_val)
        pred = solver.predict(x_col, ys)
        u_vals = pred[:, comp_u]
        v_vals = pred[:, comp_v]

        if mask_fn is not None:
            m = mask_fn({vx: x_col, vy: ys})
            u_vals = np.where(m, u_vals, np.nan)
            v_vals = np.where(m, v_vals, np.nan)

        lbl = f'x={x_val:.2f}'
        axes[0].plot(u_vals, ys, color=colors[i], lw=1.8, label=lbl)
        axes[1].plot(v_vals, ys, color=colors[i], lw=1.8, label=lbl)

        if reference_profiles is not None and x_val in reference_profiles:
            ref = reference_profiles[x_val]
            axes[0].plot(ref['u'], ref['y'], color=colors[i],
                         ls=':', lw=1.3, alpha=0.75)
            axes[1].plot(ref['v'], ref['y'], color=colors[i],
                         ls=':', lw=1.3, alpha=0.75)

    axes[0].axvline(0, color='gray', lw=0.7, alpha=0.5)
    axes[0].set_xlabel('u (горизонтальная скорость)'); axes[0].set_ylabel(vy)
    axes[0].set_title(f'{title} — профили u(y) на x=const')
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    axes[1].axvline(0, color='gray', lw=0.7, alpha=0.5)
    axes[1].set_xlabel('v (вертикальная скорость)'); axes[1].set_ylabel(vy)
    axes[1].set_title(f'{title} — профили v(y) на x=const')
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    plt.tight_layout(); plt.show(); plt.close('all')


def plot_residual_field(solver, domain, title='', N=80, mask_fn=None,
                        log_scale=True, fix_values=None):
    
    var_names = solver.var_names
    fix_values = dict(fix_values or {})
    free = [v for v in var_names if v not in fix_values]
    if len(free) != 2:
        raise ValueError(
            f'plot_residual_field работает с 2D-срезами: '
            f'найдено {len(free)} свободных переменных (free={free}). '
            f'Зафиксируй остальные через fix_values={{var: value}}.')

    vx, vy = free
    lox, hix = domain[vx]; loy, hiy = domain[vy]
    xs = np.linspace(lox, hix, N)
    ys = np.linspace(loy, hiy, N)
    X, Y = np.meshgrid(xs, ys, indexing='ij')

    coords_raw = {}
    for v in var_names:
        if v == vx:
            arr = X.ravel().reshape(-1, 1)
        elif v == vy:
            arr = Y.ravel().reshape(-1, 1)
        else:
            arr = np.full((X.size, 1), fix_values[v], dtype=np.float32)
        t = torch.tensor(arr, dtype=torch.float32,
                         device=solver.device).requires_grad_(True)
        coords_raw[v] = t

    raw = torch.cat([coords_raw[v] for v in var_names], dim=1)
    normed = solver.normalizer.normalize_tensor(raw)
    u = solver.model(normed)

    def _grad(y_c, x_c, order=1):
        r = y_c
        for _ in range(order):
            r = torch.autograd.grad(r, x_c, grad_outputs=torch.ones_like(r),
                                    create_graph=True, retain_graph=True)[0]
        return r

    residual = solver.equation(u, coords_raw, _grad)
    if isinstance(residual, (list, tuple)):
        residual_abs = torch.sqrt(sum(r ** 2 for r in residual) + 1e-30)
        label_sfx = ' (норма по компонентам системы)'
    else:
        residual_abs = torch.abs(residual)
        label_sfx = ''

    R = residual_abs.detach().cpu().numpy().reshape(X.shape)

    if mask_fn is not None:
        coords_np = {vx: X, vy: Y}
        for v, val in fix_values.items():
            coords_np[v] = np.full_like(X, val)
        m = mask_fn(coords_np)
        R = np.where(m, R, np.nan)

    fig, ax = plt.subplots(figsize=(9, 5))
    if log_scale:
        R_plot = np.log10(np.maximum(R, 1e-12))
        cf = ax.contourf(X, Y, R_plot, levels=30, cmap='hot')
        cb = plt.colorbar(cf, ax=ax)
        cb.set_label(f'log₁₀|residual|{label_sfx}')
    else:
        cf = ax.contourf(X, Y, R, levels=30, cmap='hot')
        cb = plt.colorbar(cf, ax=ax)
        cb.set_label(f'|residual|{label_sfx}')

    fix_str = ''
    if fix_values:
        fix_str = ' при ' + ', '.join(f'{v}={val:.3g}'
                                       for v, val in fix_values.items())
    ax.set_title(f'{title} — невязка PDE{fix_str}')
    ax.set_xlabel(vx); ax.set_ylabel(vy)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout(); plt.show(); plt.close('all')

    valid = R[np.isfinite(R)]
    if valid.size:
        print(f'  Max |residual|:    {valid.max():.3e}')
        print(f'  Mean |residual|:   {valid.mean():.3e}')
        print(f'  Median |residual|: {np.median(valid):.3e}')
    return R




def plot_time_probes(solver, exact_fn, domain, probe_points,
                     time_var='t', title='', N=200):
    
    t_lo, t_hi = domain[time_var]
    ts = np.linspace(t_lo, t_hi, N)
    n = len(probe_points)
    has_exact = exact_fn is not None
    ncols = 2 if has_exact else 1
    fig, axes = plt.subplots(1, ncols, figsize=(13 if has_exact else 8, 4.5))
    if ncols == 1:
        axes = [axes]

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, n))

    for i, pt in enumerate(probe_points):
        coords = {time_var: ts}
        for v in solver.var_names:
            if v == time_var:
                continue
            coords[v] = np.full_like(ts, pt[v])
        up = _predict_from_coords(solver, coords).ravel()
        label = ', '.join(f'{v}={pt[v]:.2f}'
                          for v in solver.var_names if v != time_var)

        axes[0].plot(ts, up, color=colors[i], ls='--', lw=1.6,
                     label=f'PINN: {label}')
        if has_exact:
            ue = exact_fn(**coords).ravel()
            axes[0].plot(ts, ue, color=colors[i], ls='-', lw=2.0,
                         label=f'аналит.: {label}')
            err = np.abs(up - ue)
            axes[1].semilogy(ts, np.maximum(err, 1e-16), color=colors[i],
                             lw=1.7, label=label)

    axes[0].set_xlabel(time_var); axes[0].set_ylabel('u')
    axes[0].set_title(f'{title} — u({time_var}) в точках')
    axes[0].legend(fontsize=7, ncol=1 if n <= 3 else 2)
    axes[0].grid(alpha=0.3)

    if has_exact:
        axes[1].set_xlabel(time_var); axes[1].set_ylabel('|u - u*|')
        axes[1].set_title(f'{title} — ошибка по времени (log)')
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)

    plt.tight_layout(); plt.show(); plt.close('all')


def plot_time_snapshots_2d(solver, exact_fn, domain, t_values,
                           fix_vars, spatial_vars=None,
                           time_var='t', title='', N=60):
    
    if spatial_vars is None:
        spatial_vars = [v for v in solver.var_names
                        if v != time_var and v not in fix_vars]
    assert len(spatial_vars) == 2, \
        f'Нужны 2 пространственные переменные, получено: {spatial_vars}'

    vx, vy = spatial_vars
    lox, hix = domain[vx]; loy, hiy = domain[vy]
    xs = np.linspace(lox, hix, N)
    ys = np.linspace(loy, hiy, N)
    X, Y = np.meshgrid(xs, ys, indexing='ij')

    n_t = len(t_values)
    has_exact = exact_fn is not None
    nrows = 3 if has_exact else 1
    fig = plt.figure(figsize=(4.2 * n_t, 3.5 * nrows))
    errs = []

    for i, tv in enumerate(t_values):
        coords = {vx: X, vy: Y, time_var: np.full_like(X, tv)}
        for fv, val in fix_vars.items():
            coords[fv] = np.full_like(X, val)

        up = _predict_from_coords(solver, coords)
        ax1 = fig.add_subplot(nrows, n_t, i + 1, projection='3d')
        ax1.plot_surface(X, Y, up, cmap='viridis', alpha=0.9,
                         rstride=2, cstride=2, edgecolor='none')
        ax1.set_title(f'PINN @ {time_var}={tv:.3g}')
        ax1.set_xlabel(vx); ax1.set_ylabel(vy)

        if has_exact:
            ue = exact_fn(**coords)
            ax2 = fig.add_subplot(nrows, n_t, n_t + i + 1, projection='3d')
            ax2.plot_surface(X, Y, ue, cmap='viridis', alpha=0.9,
                             rstride=2, cstride=2, edgecolor='none')
            ax2.set_title(f'аналит. @ {time_var}={tv:.3g}')
            ax2.set_xlabel(vx); ax2.set_ylabel(vy)

            ax3 = fig.add_subplot(nrows, n_t, 2 * n_t + i + 1, projection='3d')
            ax3.plot_surface(X, Y, np.abs(up - ue), cmap='plasma', alpha=0.9,
                             rstride=2, cstride=2, edgecolor='none')
            ax3.set_title(f'|Δ| @ {time_var}={tv:.3g}')
            ax3.set_xlabel(vx); ax3.set_ylabel(vy)

            errs.append(_compute_errors(up, ue))

    fixed_str = ', '.join(f'{v}={val:.2g}' for v, val in fix_vars.items())
    suptitle = f'{title} — 2D-снимки u({vx},{vy}) при {fixed_str}'
    plt.suptitle(suptitle, y=1.002)
    plt.tight_layout(); plt.show(); plt.close('all')

    if has_exact and errs:
        print(f'  Ошибки 2D-снимков:')
        for tv, e in zip(t_values, errs):
            if e['rel_defined']:
                print(f"    {time_var}={tv:.3g}: rel L2 = {e['rel_l2']:.3e}   "
                      f"max |Δ| = {e['abs_max']:.3e}")
            else:
                print(f"    {time_var}={tv:.3g}: max |Δ| = {e['abs_max']:.3e}")


def plot_error_evolution_time(solver, exact_fn, domain, time_var='t',
                              title='', n_t=50, n_space=30):
    
    t_lo, t_hi = domain[time_var]
    t_pts = np.linspace(t_lo, t_hi, n_t)
    spatial = [v for v in solver.var_names if v != time_var]

    sp_grids = [np.linspace(*domain[v], n_space) for v in spatial]
    sp_mesh = np.meshgrid(*sp_grids, indexing='ij')
    flat_sp = {v: sp_mesh[i].flatten() for i, v in enumerate(spatial)}
    n_pts = sp_mesh[0].size

    l2_rel = np.zeros(n_t)
    l_inf = np.zeros(n_t)
    rel_defined_flags = np.zeros(n_t, dtype=bool)

    for k, tval in enumerate(t_pts):
        coords = dict(flat_sp)
        coords[time_var] = np.full(n_pts, tval)
        up = _predict_from_coords(solver, coords).ravel()
        ue = exact_fn(**coords).ravel()

        e = _compute_errors(up, ue)
        l2_rel[k] = e['rel_l2']
        l_inf[k] = e['abs_max']
        rel_defined_flags[k] = e['rel_defined']

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    
    mask_ok = rel_defined_flags
    if mask_ok.any():
        axes[0].semilogy(t_pts[mask_ok], l2_rel[mask_ok],
                         'b-', lw=2.0, label='Rel L2')
    
    mask_bad = ~rel_defined_flags
    if mask_bad.any():
        axes[0].semilogy(t_pts[mask_bad], l2_rel[mask_bad],
                         'r--', lw=1.2, alpha=0.6,
                         label='Rel L2 (аналит.≈0, =max |Δ|)')
    axes[0].set_xlabel(time_var)
    axes[0].set_ylabel('относительная L2')
    axes[0].set_title(f'{title} — эволюция Rel L2 ошибки')
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].semilogy(t_pts, l_inf, 'k-', lw=2.0, label='max |Δ|')
    axes[1].set_xlabel(time_var)
    axes[1].set_ylabel('L∞ абсолютная')
    axes[1].set_title(f'{title} — эволюция max абс. ошибки')
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.suptitle(f'{title} — ошибка во времени', fontsize=12, y=1.02)
    plt.tight_layout(); plt.show(); plt.close('all')

    print(f'  Сводка по времени:')
    print(f'    min Rel L2: {l2_rel[mask_ok].min():.3e}' if mask_ok.any()
          else '    (Rel L2 нигде не определена)')
    print(f'    max Rel L2: {l2_rel[mask_ok].max():.3e}' if mask_ok.any() else '')
    print(f'    min max|Δ|: {l_inf.min():.3e}')
    print(f'    max max|Δ|: {l_inf.max():.3e}')
