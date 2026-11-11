"""Quantitative hardware-noise analysis of the Aquila features.

References (`results/emulation/`):
  sv     ideal statevector, exact readout times
  hw     hardware-protocol statevector: a fresh program from |0> per readout time, with the
         submitted waveforms (what Aquila was asked to do), exact expectation values
  shot   shot emulation of `hw`: the first N stored samples, N = the number of Aquila records
         kept for the same datapoint and readout (matched statistics)
  aquila every kept Aquila record (as in the published features)
  sv_pub statevector features with readouts at the integrator's step endpoints

Outputs (results/hardware_noise/):
  agreement_per_sample.csv    Pearson, Spearman, RMSE, MAE across observables within a sample;
                              median and IQR over samples, per readout and observable type
  agreement_per_feature.csv   Pearson across samples per feature; median and IQR over features
  retention.csv               signal retention a(T): slope of the centred Aquila features on the
                              centred reference, pooled over features, with bootstrap CI
  decay_fit.csv               a(T) = exp(-T / tau) and median-RMSE growth fits
  bias_variance_curves.csv    MSE(s) vs the reference for Aquila and shot-emulation subsamples
  bias_variance_fit.csv       MSE(s) = b^2 + sigma^2 / s fits, direct estimates, crossover s*
  error_decomposition.csv     b^2 split into per-feature offset, attenuation and residual

    python analysis/hardware_noise/analysis.py
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

from common import DATASETS, OUT, READOUT_TIMES, Records, blocks, load_features, n_obs, obs_types, records_features

SEED = 20261021
REPS = 5           # random subsamples per shot count
BOOT = 500         # bootstrap resamples over datapoints


def matched_shot_features(hwemu, ids, counts):
    """Shot emulation with, per datapoint and readout, as many samples as Aquila kept."""
    out = np.empty((5, n_obs(hwemu.nsites), len(ids)))
    from common import observables
    for c, id_ in enumerate(ids):
        for r in range(1, 6):
            out[r - 1, :, c] = observables(hwemu.spins(id_, r)[: counts.loc[id_, r]])
    return out


def load_dataset(name):
    d = DATASETS[name]
    A, H = Records(d["aquila"]), Records(d["hwemu"])
    ids = A.ids
    k = A.nsites
    counts = A.counts()
    data = {
        "aquila": blocks(records_features(A, ids), k),
        "shot": matched_shot_features(H, ids, counts),
        "hw": blocks(load_features(d["hw"], ids), k),
        "sv": blocks(load_features(d["sv"], ids), k),
        "sv_pub": blocks(load_features(d["sv_integrator"], ids), k),
    }
    return dict(name=name, ids=ids, k=k, A=A, H=H, counts=counts, data=data)


def type_masks(k):
    t = obs_types(k)
    return {"Z": t == "Z", "ZZ": t == "ZZ", "all": np.ones(len(t), bool)}


def colwise_pearson(X, Y):
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    den = np.sqrt((Xc ** 2).sum(0) * (Yc ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (Xc * Yc).sum(0) / den


def iqr_summary(v):
    v = v[np.isfinite(v)]
    return dict(median=np.median(v), q25=np.quantile(v, .25), q75=np.quantile(v, .75), mean=v.mean(), n=len(v))


COMPARISONS = [("aquila", "hw"), ("aquila", "sv"), ("aquila", "shot"), ("shot", "hw"), ("shot", "sv"),
               ("hw", "sv"), ("aquila", "sv_pub")]


def agreement(ds, sel, subset):
    """Per-sample and per-feature agreement tables."""
    rows_s, rows_f = [], []
    for x, y in COMPARISONS:
        X, Y = ds["data"][x][:, :, sel], ds["data"][y][:, :, sel]
        for ti, T in enumerate(READOUT_TIMES):
            for typ, m in type_masks(ds["k"]).items():
                Xt, Yt = X[ti][m], Y[ti][m]          # observables × samples
                d = Xt - Yt
                per = {
                    "pearson": colwise_pearson(Xt, Yt),
                    "spearman": colwise_pearson(stats.rankdata(Xt, axis=0), stats.rankdata(Yt, axis=0)),
                    "rmse": np.sqrt((d ** 2).mean(0)),
                    "mae": np.abs(d).mean(0),
                }
                for metric, v in per.items():
                    rows_s.append(dict(encoding=ds["name"], subset=subset, x=x, y=y, readout_us=T, obs=typ,
                                       metric=metric, **iqr_summary(v)))
                pooled = np.corrcoef(Xt.ravel(), Yt.ravel())[0, 1]
                rows_s.append(dict(encoding=ds["name"], subset=subset, x=x, y=y, readout_us=T, obs=typ,
                                   metric="pearson_pooled", median=pooled, n=Xt.size))
                rf = colwise_pearson(Xt.T, Yt.T)      # per feature, across samples
                rows_f.append(dict(encoding=ds["name"], subset=subset, x=x, y=y, readout_us=T, obs=typ,
                                   metric="pearson_across_samples", **iqr_summary(rf)))
    return rows_s, rows_f


def retention_slope(X, Y):
    """Slope of centred X on centred Y (centring per feature across samples), pooled over features."""
    Xc = X - X.mean(1, keepdims=True)
    Yc = Y - Y.mean(1, keepdims=True)
    return (Xc * Yc).sum() / (Yc ** 2).sum()


def retention(ds, sel, subset, rng):
    rows = []
    n = sel.sum()
    boot_idx = [rng.integers(0, n, n) for _ in range(BOOT)]
    for x, y in [("aquila", "hw"), ("aquila", "sv"), ("shot", "hw"), ("hw", "sv")]:
        X, Y = ds["data"][x][:, :, sel], ds["data"][y][:, :, sel]
        for ti, T in enumerate(READOUT_TIMES):
            for typ, m in type_masks(ds["k"]).items():
                Xt, Yt = X[ti][m], Y[ti][m]
                a = retention_slope(Xt, Yt)
                bs = [retention_slope(Xt[:, b], Yt[:, b]) for b in boot_idx]
                sig = Yt.var(1).mean()           # mean across-sample variance of the reference features
                rows.append(dict(encoding=ds["name"], subset=subset, x=x, y=y, readout_us=T, obs=typ, slope=a,
                                 ci_lo=np.quantile(bs, .025), ci_hi=np.quantile(bs, .975),
                                 ref_signal_var=sig))
    return rows


def expdecay(T, tau):
    return np.exp(-T / tau)


def expdecay_amp(T, a0, tau):
    return a0 * np.exp(-T / tau)


def decay_fits(ret, agr):
    rows = []
    for (enc, subset, x, y, typ), g in ret.groupby(["encoding", "subset", "x", "y", "obs"]):
        if (x, y) not in [("aquila", "hw"), ("aquila", "sv")]:
            continue
        g = g.sort_values("readout_us")
        T, a = g.readout_us.to_numpy(), g.slope.to_numpy()
        sd = np.maximum((g.ci_hi - g.ci_lo).to_numpy() / 3.92, 1e-3)
        for model, f, p0 in [("exp(-T/tau)", expdecay, [2.0]), ("A exp(-T/tau)", expdecay_amp, [1.0, 2.0])]:
            try:
                p, cov = curve_fit(f, T, a, p0=p0, sigma=sd, absolute_sigma=False, maxfev=10000)
                tau, tau_se = p[-1], np.sqrt(np.diag(cov))[-1]
                rows.append(dict(encoding=enc, subset=subset, x=x, y=y, obs=typ, quantity="retention slope",
                                 model=model, A=p[0] if len(p) == 2 else 1.0, tau_us=tau, tau_se=tau_se,
                                 rmse_fit=np.sqrt(np.mean((f(T, *p) - a) ** 2))))
            except RuntimeError as e:
                rows.append(dict(encoding=enc, subset=subset, x=x, y=y, obs=typ, quantity="retention slope",
                                 model=model, note=str(e)))
    sel = agr[(agr.metric == "pearson") & (agr.x == "aquila") & agr.y.isin(["hw", "sv"])]
    for (enc, subset, x, y, typ), g in sel.groupby(["encoding", "subset", "x", "y", "obs"]):
        g = g.sort_values("readout_us")
        T, r = g.readout_us.to_numpy(), g["median"].to_numpy()
        try:
            p, cov = curve_fit(expdecay_amp, T, r, p0=[1.0, 2.0], maxfev=10000)
            rows.append(dict(encoding=enc, subset=subset, x=x, y=y, obs=typ, quantity="median per-sample Pearson",
                             model="A exp(-T/tau)", A=p[0], tau_us=p[1], tau_se=np.sqrt(cov[1, 1]),
                             rmse_fit=np.sqrt(np.mean((expdecay_amp(T, *p) - r) ** 2))))
        except RuntimeError as e:
            rows.append(dict(encoding=enc, subset=subset, x=x, y=y, obs=typ, quantity="median per-sample Pearson",
                             model="A exp(-T/tau)", note=str(e)))
    return rows


def bias_variance(ds, sel_ids, subset, grid, rng):
    """MSE(s) of subsampled Aquila and shot-emulation features against hw and sv."""
    k = ds["k"]
    masks = type_masks(k)
    ids = ds["ids"][sel_ids]
    curves = []
    refs = {r: ds["data"][r][:, :, sel_ids] for r in ("hw", "sv")}
    for src, rec in [("aquila", ds["A"]), ("shot", ds["H"])]:
        for s in grid:
            sq = {r: np.zeros((5, n_obs(k))) for r in refs}
            for _ in range(REPS):
                F = records_features(rec, ids, n=s, rng=rng).reshape(5, n_obs(k), len(ids))
                for r, R in refs.items():
                    sq[r] += ((F - R) ** 2).mean(2) / REPS
            for r in refs:
                for ti, T in enumerate(READOUT_TIMES):
                    for typ, m in masks.items():
                        curves.append(dict(encoding=ds["name"], subset=subset, source=src, ref=r, shots=s,
                                           readout_us=T, obs=typ, mse=sq[r][ti][m].mean()))
        # all readouts together
    curves = pd.DataFrame(curves)
    allT = (curves.groupby(["encoding", "subset", "source", "ref", "shots", "obs"], as_index=False).mse.mean()
            .assign(readout_us="all"))
    curves = pd.concat([curves, allT], ignore_index=True)

    # direct estimates from all records: b^2 = (xbar - ref)^2 - S^2/N, sigma^2 = mean S^2
    N = ds["counts"].loc[ids].to_numpy().T[:, None, :]          # readout × 1 × samples
    direct = []
    for src in ("aquila", "shot"):
        X = ds["data"][src][:, :, sel_ids]
        S2 = (1 - X ** 2) * N / (N - 1)
        for r, R in refs.items():
            b2 = (X - R) ** 2 - S2 / N
            for ti, T in list(enumerate(READOUT_TIMES)) + [(None, "all")]:
                for typ, m in masks.items():
                    sl = (slice(None) if ti is None else ti)
                    direct.append(dict(encoding=ds["name"], subset=subset, source=src, ref=r, readout_us=T, obs=typ,
                                       b2_direct=b2[sl][..., m, :].mean() if ti is None else b2[ti][m].mean(),
                                       sigma2_direct=S2[sl][..., m, :].mean() if ti is None else S2[ti][m].mean()))
    return curves, pd.DataFrame(direct)


def fit_bias_variance(curves, direct):
    rows = []
    for key, g in curves.groupby(["encoding", "subset", "source", "ref", "readout_us", "obs"]):
        x = 1.0 / g.shots.to_numpy(float)
        y = g.mse.to_numpy()
        # weights ~ 1/expected MSE so that the large-s points, which fix b^2, count
        A = np.vstack([np.ones_like(x), x]).T
        w = 1.0 / y
        (b2, s2), *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
        enc, subset, src, ref, T, typ = key
        dr = direct[(direct.encoding == enc) & (direct.subset == subset) & (direct.source == src) &
                    (direct.ref == ref) & (direct.readout_us.astype(str) == str(T)) & (direct.obs == typ)].iloc[0]
        rows.append(dict(encoding=enc, subset=subset, source=src, ref=ref, readout_us=T, obs=typ,
                         b2_fit=b2, sigma2_fit=s2, b2_direct=dr.b2_direct, sigma2_direct=dr.sigma2_direct,
                         s_star=s2 / b2 if b2 > 0 else np.inf,
                         s_star_direct=dr.sigma2_direct / dr.b2_direct if dr.b2_direct > 0 else np.inf,
                         mse_at_240_over_b2=(dr.b2_direct + dr.sigma2_direct / 240) / dr.b2_direct
                         if dr.b2_direct > 0 else np.inf,
                         max_shots=int(g.shots.max())))
    return pd.DataFrame(rows)


def error_decomposition(ds, sel, subset):
    """Mean squared Aquila-vs-reference error split into: shot noise (S^2/N), per-feature offset
    (mean over samples), attenuation ((a-1)^2 var(ref), a = per-feature slope) and the residual
    sample-dependent hardware error."""
    rows = []
    N = ds["counts"].to_numpy().T[:, None, sel]
    for src in ("aquila", "shot"):
        X = ds["data"][src][:, :, sel]
        for ref in ("hw", "sv"):
            R = ds["data"][ref][:, :, sel]
            for ti, T in enumerate(READOUT_TIMES):
                for typ, m in type_masks(ds["k"]).items():
                    Xt, Rt, Nt = X[ti][m], R[ti][m], N[ti][0]
                    total = ((Xt - Rt) ** 2).mean()
                    shot = ((1 - Xt ** 2) / (Nt - 1)).mean()
                    offset = ((Xt.mean(1) - Rt.mean(1)) ** 2).mean()
                    Xc, Rc = Xt - Xt.mean(1, keepdims=True), Rt - Rt.mean(1, keepdims=True)
                    vr = (Rc ** 2).mean(1)
                    a = (Xc * Rc).mean(1) / np.where(vr > 0, vr, np.nan)
                    a = np.nan_to_num(a)
                    atten = ((a - 1) ** 2 * vr).mean()
                    resid = total - offset - atten - shot
                    # what a standardising classifier sees: retained signal a^2 var(ref) against the
                    # sample-dependent error (residual hardware error + shot noise sigma^2 / s)
                    retained = (a ** 2 * vr).mean()
                    sigma2 = (1 - Xt ** 2).mean() * Nt.mean() / (Nt.mean() - 1)
                    rows.append(dict(encoding=ds["name"], subset=subset, source=src, ref=ref, readout_us=T, obs=typ,
                                     mse_total=total, shot_noise=shot, offset=offset, attenuation=atten,
                                     residual=resid, ref_signal_var=vr.mean(), median_feature_slope=np.median(a),
                                     retained_signal=retained, sigma2=sigma2,
                                     s_star_residual=sigma2 / resid if resid > 0 else np.inf,
                                     snr_ref_240=vr.mean() / (sigma2 / 240),
                                     snr_240=retained / (resid + sigma2 / 240),
                                     snr_inf=retained / resid if resid > 0 else np.inf))
    return rows


def main():
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    det, pos = load_dataset("detuning"), load_dataset("position")
    c1000 = set(pos["ids"])
    counts_det = det["counts"]
    fig8 = counts_det.index[counts_det.min(1) > 500].to_numpy()       # the 10-datapoint ~1000-shot run
    assert len(fig8) == 10
    subsets = [
        (det, np.ones(len(det["ids"]), bool), "CARDS_2500"),
        (det, np.isin(det["ids"], list(c1000)), "CARDS_1000"),
        (det, np.isin(det["ids"], fig8), "fig8_10"),
        (pos, np.ones(len(pos["ids"]), bool), "CARDS_1000"),
    ]
    agr_s, agr_f, ret, dec = [], [], [], []
    for ds, sel, name in subsets:
        s, f = agreement(ds, sel, name)
        agr_s += s
        agr_f += f
        ret += retention(ds, sel, name, rng)
        dec += error_decomposition(ds, sel, name)
        print(f"agreement/retention done: {ds['name']} {name}")
    agr_s, agr_f, ret = pd.DataFrame(agr_s), pd.DataFrame(agr_f), pd.DataFrame(ret)
    agr_s.to_csv(OUT / "agreement_per_sample.csv", index=False)
    agr_f.to_csv(OUT / "agreement_per_feature.csv", index=False)
    ret.to_csv(OUT / "retention.csv", index=False)
    pd.DataFrame(dec).to_csv(OUT / "error_decomposition.csv", index=False)
    pd.DataFrame(decay_fits(ret, agr_s)).to_csv(OUT / "decay_fit.csv", index=False)

    grids = {
        "CARDS_2500": [5, 10, 20, 40, 80, 120, 160, 198],
        "CARDS_1000": [5, 10, 20, 40, 80, 120, 124],
        "fig8_10": [5, 10, 20, 40, 80, 160, 320, 640, 931],
    }
    curves, direct = [], []
    for ds, sel, name in subsets:
        grid = [s for s in grids[name] if s <= ds["counts"].to_numpy()[sel].min()]
        c, d = bias_variance(ds, sel, name, grid, rng)
        curves.append(c)
        direct.append(d)
        print(f"bias-variance done: {ds['name']} {name} grid {grid}")
    curves, direct = pd.concat(curves), pd.concat(direct)
    curves.to_csv(OUT / "bias_variance_curves.csv", index=False)
    direct.to_csv(OUT / "bias_variance_direct.csv", index=False)
    fit_bias_variance(curves, direct).to_csv(OUT / "bias_variance_fit.csv", index=False)


if __name__ == "__main__":
    main()
