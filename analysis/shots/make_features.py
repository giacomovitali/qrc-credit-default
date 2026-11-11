"""Reservoir features at reduced shot counts, and their error against the noiseless references.

Sources (per encoding, from the record export (export_bits.jl)):
- emu    — hardware-protocol shot emulation (1,000 i.i.d. records per readout); the first s
           records, as `reduce_bitstrings.jl` does, so the counts are nested;
- aquila — kept Aquila records; a seeded random permutation per (datapoint, readout), then the
           first min(s, kept) records, also nested. Where fewer than s records were kept, all are
           used and the cap is counted in `capped_frac`.

Writes `results/shots/features/<enc>_<source>_s<s>.csv` (one column per datapoint, named by ID,
as the reservoir outputs) with a manifest, and `results/shots/feature_error.csv`.

    python analysis/shots/make_features.py [--seed 6]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hardware_noise"))
from common import DATASETS, ROOT, Records, blocks, load_features, n_obs, observables, obs_types  # noqa: E402

OUT = ROOT / "results" / "shots"
SHOTS = [10, 20, 40, 80, 120, 160, 200, 240, 400, 1000]
AQUILA_SHOTS = [10, 20, 40, 80, 120, 160, 200, 240]  # ≤ the 240 requested per readout


def git_state():
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip())
    return sha, dirty


def nested_features(rec, ids, shots, rng=None):
    """{s: features × datapoints} and {s: fraction of (datapoint, readout) with < s records}."""
    F = {s: np.empty((5 * n_obs(rec.nsites), len(ids))) for s in shots}
    capped = {s: 0 for s in shots}
    for c, id_ in enumerate(ids):
        per_s = {s: [] for s in shots}
        for r in range(1, 6):
            z = rec.spins(id_, r)
            if rng is not None:
                z = z[rng.permutation(len(z))]
            for s in shots:
                capped[s] += len(z) < s
                per_s[s].append(observables(z[:s]))
        for s in shots:
            F[s][:, c] = np.concatenate(per_s[s])
    return F, {s: capped[s] / (5 * len(ids)) for s in shots}


def error_rows(F, ref, nsites, **keys):
    """MSE and median per-sample Pearson r per readout and observable type."""
    t = obs_types(nsites)
    A, B = blocks(F, nsites), blocks(ref, nsites)
    rows = []
    for r in range(5):
        for typ in ("Z", "ZZ", "all"):
            m = np.ones(len(t), bool) if typ == "all" else t == typ
            a, b = A[r, m], B[r, m]
            d = a - b
            ac, bc = a - a.mean(0), b - b.mean(0)
            pr = (ac * bc).sum(0) / np.sqrt((ac ** 2).sum(0) * (bc ** 2).sum(0))
            # per feature, across samples (what a classifier sees)
            af, bf = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
            pf = (af * bf).sum(1) / np.sqrt((af ** 2).sum(1) * (bf ** 2).sum(1))
            rows.append(dict(keys, readout_us=0.5 * (r + 1), obs=typ, mse=float((d ** 2).mean()),
                             rmse_median=float(np.median(np.sqrt((d ** 2).mean(0)))),
                             r_sample_median=float(np.nanmedian(pr)),
                             r_feature_median=float(np.nanmedian(pf))))
    return rows


def write(F, ids, path, manifest):
    pd.DataFrame(F, columns=[str(i) for i in ids]).to_csv(path, index=False, float_format="%.10g")
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=6, help="seed of the Aquila record permutation")
    args = ap.parse_args()
    (OUT / "features").mkdir(parents=True, exist_ok=True)
    sha, dirty = git_state()
    rows, checks = [], []
    for enc, d in DATASETS.items():
        aq = Records(d["aquila"])
        hw = Records(d["hwemu"])
        ids = aq.ids  # CARDS_2500 (detuning) / CARDS_1000 (position)
        sv, model = load_features(d["sv"], ids), load_features(d["hw"], ids)
        stem = Path(d["hw"]).name.replace("_hw.csv", "")
        Fe, _ = nested_features(hw, ids, SHOTS)
        Fa, cap = nested_features(aq, ids, AQUILA_SHOTS, np.random.default_rng(args.seed))
        Fall = nested_features(aq, ids, [10 ** 6])[0][10 ** 6]  # every kept record

        # Regression: emulation at 240/1000 = the reduce_bitstrings outputs; all records = published.
        for s in (240, 1000):
            ref = load_features(ROOT / "results" / "emulation" / f"{stem}_hwshots_{s}.csv", ids)
            checks.append(dict(encoding=enc, check=f"emu s={s} vs {stem}_hwshots_{s}.csv",
                               max_abs_diff=float(np.abs(Fe[s] - ref).max())))
        pub = load_features(d["published"], ids)
        dcol = np.abs(Fall - pub).max(0)
        checks.append(dict(encoding=enc, check="aquila all records vs published hardware features",
                           max_abs_diff=float(np.sort(dcol)[-11] if enc == "detuning" else dcol.max()),
                           note="detuning: max over all but the 10 Fig-8 datapoints" if enc == "detuning" else ""))

        common = dict(code_commit=sha, dirty=dirty, encoding=enc, ids=len(ids),
                      script="analysis/shots/make_features.py")
        for s, F in Fe.items():
            write(F, ids, OUT / "features" / f"{enc}_emu_s{s}.csv",
                  dict(common, source=f"first {s} records of {d['hwemu']} (hardware-protocol shot emulation)"))
        for s, F in Fa.items():
            write(F, ids, OUT / "features" / f"{enc}_aquila_s{s}.csv",
                  dict(common, source=f"Aquila kept records {d['aquila']}, permutation seed {args.seed}, "
                                      f"first min({s}, kept)", capped_frac=cap[s]))

        kept = aq.counts().to_numpy()
        for name, F, src, s, c in ([("emu", Fe[s], "emu", s, 0.0) for s in SHOTS]
                                   + [("aquila", Fa[s], "aquila", s, cap[s]) for s in AQUILA_SHOTS]
                                   + [("aquila", Fall, "aquila", -1, 0.0)]):
            for refname, ref in (("sv", sv), ("model", model)):
                rows += error_rows(F, ref, d["nsites"], encoding=enc, source=src, shots=s,
                                   capped_frac=c, reference=refname)
        for refname, ref in (("sv", sv),):
            rows += error_rows(model, ref, d["nsites"], encoding=enc, source="model", shots=0,
                               capped_frac=0.0, reference=refname)
        print(enc, "kept records per readout: median", np.median(kept), "min", kept.min(),
              "| capped fraction", {s: round(v, 3) for s, v in cap.items()}, flush=True)

    pd.DataFrame(rows).to_csv(OUT / "feature_error.csv", index=False)
    pd.DataFrame(checks).to_csv(OUT / "regression.csv", index=False)
    print(pd.DataFrame(checks).to_string())


if __name__ == "__main__":
    main()
