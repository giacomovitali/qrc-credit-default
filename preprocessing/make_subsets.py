"""Independent stratified subsamples of CARDS_30000 and the balance of the actual QPU subsets.

Subsamples. The QPU subset CARDS_2500 is the test side of
    train_test_split(cleaned 29,601, test_size=2571, random_state=7, stratify=label)
and CARDS_1000 the test side of the same call on CARDS_2500 with test_size=1000. Both are first
reproduced here from the raw UCI file (checked against the distributed label files, IDs and order). The
K = 10 subsamples per size are drawn with the same call on the cleaned 29,601, independently of each other
and of the actual subsets: size 2,571 with random_state 101-110, size 1,000 directly from the 29,601 (not
from a 2,571 subsample) with random_state 201-210. The two sizes need disjoint seeds: at equal
random_state, sklearn's stratified split puts the smaller test side inside the larger one.

For each subsample <tag> (n2571s01 ... n1000s10) it writes to results/subsets/subsets/:
  <tag>_labels.csv   ID, default_payment_next_month, in draw order
  <tag>_fc12.csv     the 12 aggregated features (rows, published files)
  <tag>_fc24.csv     the 24 preprocessed variables (rows)
  <tag>_crc.csv      CRC features (one column per ID, published files)
  <tag>_dethw.csv    QRC detuning, hardware-protocol emulation (one column per ID)
  <tag>_poshw.csv    size 1,000 only: QRC position, hardware-protocol emulation, once the emulation of
                     position_input/pos1000.csv (below) exists
These are column/row selections of existing files, values unchanged (checked on read-back). The actual
subsets are written the same way as tags cards2500 / cards1000, as a regression check of this path against
the baseline/benchmark runs.

Position. Only CARDS_1000 has position features, so the union of the size-1,000 subsamples (12 features,
rows copied verbatim) goes to position_input/pos1000.csv, next to a link to qpu_samples.csv (the scale
reference of configs/reservoir/position_hardware.toml; every set here has the same global minimum, so the encoding is
the one of CARDS_1000). After the emulation (run_position.sh) the IDs shared with CARDS_1000 are compared
with the published features.

Balance. For CARDS_2500, CARDS_1000 and every subsample: the class-1 fraction, and per feature the
standardised mean difference against the rest of the 29,601 ((mean_S - mean_rest) / sqrt((var_S +
var_rest) / 2)) and the two-sample KS test against the rest, on the 12 aggregated features (the model
inputs) and the 23 cleaned raw attributes. The random subsamples give the reference distribution of these
statistics under stratified random selection.

Writes results/subsets/{subsets.csv, overlap.csv, balance.csv, balance_summary.csv, subsets_manifest.json}.

    python preprocessing/make_subsets.py
"""
import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "subsets"
SUB = OUT / "subsets"
D, E, F = ROOT / "dataset_aquila", ROOT / "results" / "emulation", ROOT / "results" / "baselines" / "features"
RAW = D / "default_of_credit_card_clients.csv"
LABEL = "default_payment_next_month"
SIZES = {2571: range(101, 111), 1000: range(201, 211)}
SOURCES = {  # pipeline input -> (QPU-group file, emulation-group file, layout)
    "fc12": (D / "qpu_samples.csv", D / "emulation_samples.csv", "rows"),
    "fc24": (F / "qpu_samples_24.csv", F / "emulation_samples_24.csv", "rows"),
    "crc": (D / "qpu_samples_crc_labeled.csv", D / "emulation_samples_crc_labeled.csv", "columns"),
    "dethw": (E / "detuning_encoding_qpu_samples_12_delta_7.6_deltamax_-9.0_hw.csv",
              E / "detuning_encoding_emulation_samples_12_delta_7.6_deltamax_-9.0_hw.csv", "columns"),
}
POS_IN = OUT / "position_input"
POS_PUBLISHED = E / "position_encoding_qpu_samples_pos_12_delta_2.0_lambda_1.0_hw.csv"
POS_NEW = OUT / "emulation" / "position_encoding_pos1000_12_delta_2.0_lambda_1.0_hw.csv"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cleaned():
    # the preprocessing utils.py, imported by path (classification/ has another utils.py)
    spec = importlib.util.spec_from_file_location("prep_utils", ROOT / "preprocessing" / "utils.py")
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    return utils.clean_dataset(pd.read_csv(RAW, index_col="ID"))


def draw(df, size, seed):
    return train_test_split(df, test_size=size, random_state=seed, stratify=df[LABEL])[1]


def check_actual(df):
    qpu = draw(df, 2571, 7)
    c1000 = draw(qpu, 1000, 7)
    for got, path in [(qpu, D / "labels_qpu.csv"), (c1000, D / "labels_qpu_pos.csv")]:
        ref = pd.read_csv(path)
        if not (got.index.to_numpy() == ref.ID.to_numpy()).all() or not (got[LABEL].to_numpy() == ref[LABEL].to_numpy()).all():
            raise SystemExit(f"the stratified draw does not reproduce {path.name}")
    return {"CARDS_2500": qpu, "CARDS_1000": c1000}


def read_tokens(path, layout):
    """The file's CSV tokens, keyed by datapoint ID: (header, {ID: tokens}) for rows, (n_rows, {ID: column
    tokens}) for columns. Tokens are copied verbatim, so the subsets parse to exactly the same floats
    (re-serialising through pandas does not round-trip with its default parser)."""
    lines = path.read_text().splitlines()
    head = lines[0].split(",")
    if layout == "rows":
        i = head.index("ID")
        return head, {int(t[i]): t for t in (ln.split(",") for ln in lines[1:])}
    body = [ln.split(",") for ln in lines[1:]]
    off = len(body[0]) - len(head)          # the CRC files carry an unnamed row index (pandas' implicit index)
    return len(body), {int(c): [r[j + off] for r in body] for j, c in enumerate(head) if c.isdigit()}


def write_subset(path, head, tokens, ids, layout):
    if layout == "rows":
        text = [",".join(head)] + [",".join(tokens[i]) for i in ids]
    else:
        cols = [tokens[i] for i in ids]
        text = [",".join(map(str, ids))] + [",".join(c[r] for c in cols) for r in range(head)]
    path.write_text("\n".join(text) + "\n")


def same_values(path, sources, ids, layout):
    """Read-back check: the subset file parses to the same floats as the source files."""
    if layout == "rows":
        ref = pd.concat([pd.read_csv(p, index_col="ID") for p in sources])
        return np.array_equal(pd.read_csv(path, index_col="ID").to_numpy(), ref.loc[ids].to_numpy())
    want = set(ids)
    ref = pd.concat([pd.read_csv(p, usecols=lambda c: c.isdigit() and int(c) in want) for p in sources], axis=1)
    return np.array_equal(pd.read_csv(path).to_numpy(), ref[[str(i) for i in ids]].to_numpy())


def write_features(tags, subs):
    for name, (qpu, emu, layout) in SOURCES.items():
        (head, a), (_, b) = read_tokens(qpu, layout), read_tokens(emu, layout)
        tokens = a | b
        for tag in tags:
            ids = list(subs[tag].index)
            write_subset(SUB / f"{tag}_{name}.csv", head, tokens, ids, layout)
        if not same_values(SUB / f"{tag}_{name}.csv", (qpu, emu), ids, layout):
            raise SystemExit(f"{tag}_{name}.csv: values differ from the source files")
        print(f"wrote {name} for {len(tags)} subsets (read-back identical)")
        if name == "fc12":
            POS_IN.mkdir(exist_ok=True)
            union = sorted({i for t in tags if t.startswith("n1000s") for i in subs[t].index})
            write_subset(POS_IN / "pos1000.csv", head, tokens, union, layout)
            link = POS_IN / "qpu_samples.csv"
            if not link.exists():
                link.symlink_to(os.path.relpath(qpu, POS_IN))
            print(f"position input: {len(union)} samples")


def write_position(tags, subs):
    """Per size-1,000 subsample position features, from the subset emulation (analysis/subsets/run_position.sh); check against the published ones."""
    if not POS_NEW.exists():
        print(f"no {POS_NEW.name} yet: run analysis/subsets/run_position.sh, then this script again")
        return None
    new = pd.read_csv(POS_NEW)
    pub = pd.read_csv(POS_PUBLISHED)
    shared = sorted(set(new.columns) & set(pub.columns), key=int)
    diff = float(np.abs(new[shared].to_numpy() - pub[shared].to_numpy()).max())
    print(f"position: {len(shared)} IDs shared with CARDS_1000, max |new - published| = {diff:.3g}")
    head, tokens = read_tokens(POS_NEW, "columns")
    for tag in [t for t in tags if t.startswith("n1000s")]:
        write_subset(SUB / f"{tag}_poshw.csv", head, tokens, list(subs[tag].index), "columns")
    if not same_values(SUB / f"{tag}_poshw.csv", (POS_NEW,), list(subs[tag].index), "columns"):
        raise SystemExit(f"{tag}_poshw.csv: values differ from the source file")
    return {"n_shared_with_cards1000": len(shared), "max_abs_diff_vs_published": diff,
            "sha256": sha(POS_NEW)}


def balance(df, sets):
    agg = pd.concat([pd.read_csv(p, index_col="ID") for p in SOURCES["fc12"][:2]])
    agg.index = agg.index.astype(int)
    agg.columns = [f"f{c}" for c in agg.columns]
    raw = df.drop(columns=LABEL)
    rows = []
    for name, sub in sets.items():
        inside = df.index.isin(sub.index)
        for kind, X in [("aggregated12", agg.loc[df.index]), ("raw23", raw)]:
            for col in X.columns:
                a, b = X.loc[inside, col].to_numpy(float), X.loc[~inside, col].to_numpy(float)
                pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
                ks = ks_2samp(a, b)
                rows.append(dict(subset=name, kind=kind, feature=col, smd=(a.mean() - b.mean()) / pooled,
                                 ks_stat=ks.statistic, ks_p=ks.pvalue))
    bal = pd.DataFrame(rows)
    summ = []
    for (name, kind), g in bal.groupby(["subset", "kind"], sort=False):
        p = np.sort(g.ks_p.to_numpy())
        m = len(p)
        holm_min = min(1.0, p[0] * m)
        summ.append(dict(subset=name, kind=kind, n=len(sets[name]), class1=sets[name][LABEL].mean(),
                         max_abs_smd=g.smd.abs().max(), mean_abs_smd=g.smd.abs().mean(),
                         max_ks=g.ks_stat.max(), min_ks_p=p[0], min_ks_p_holm=holm_min,
                         n_ks_p_below_005=int((p < 0.05).sum())))
    summ = pd.DataFrame(summ)
    # where the actual subsets fall among the random ones of the same size (fraction of subsamples with a
    # smaller statistic: high = less balanced than typical)
    for actual, size in [("CARDS_2500", 2571), ("CARDS_1000", 1000)]:
        for kind in ["aggregated12", "raw23"]:
            ref = summ[summ.subset.str.startswith(f"n{size}s") & (summ.kind == kind)]
            i = summ.index[(summ.subset == actual) & (summ.kind == kind)][0]
            for col in ["max_abs_smd", "mean_abs_smd", "max_ks"]:
                summ.loc[i, f"{col}_frac_random_below"] = (ref[col] < summ.loc[i, col]).mean()
    return bal, summ


def main():
    SUB.mkdir(parents=True, exist_ok=True)
    df = cleaned()
    actual = check_actual(df)
    print(f"cleaned {len(df)}; CARDS_2500 and CARDS_1000 reproduced from {RAW.name}")
    qpu_ids = set(actual["CARDS_2500"].index)
    subs = {f"n{size}s{k:02d}": draw(df, size, seed)
            for size, seeds in SIZES.items() for k, seed in enumerate(seeds, 1)}
    seeds = {f"n{size}s{k:02d}": seed for size, ss in SIZES.items() for k, seed in enumerate(ss, 1)}

    rows = []
    for tag, sub in subs.items():
        sub[[LABEL]].to_csv(SUB / f"{tag}_labels.csv")
        rows += [dict(subset=tag, size=len(sub), random_state=seeds[tag], ID=i, label=y,
                      group="qpu" if i in qpu_ids else "emulation") for i, y in sub[LABEL].items()]
    pd.DataFrame(rows).to_csv(OUT / "subsets.csv", index=False)

    every = {**actual, **subs}
    names = list(every)
    ov = pd.DataFrame([[len(set(every[a].index) & set(every[b].index)) for b in names] for a in names],
                      index=names, columns=names)
    ov.to_csv(OUT / "overlap.csv")

    regression = {"cards2500": actual["CARDS_2500"], "cards1000": actual["CARDS_1000"]}
    for tag, sub in regression.items():
        sub[[LABEL]].to_csv(SUB / f"{tag}_labels.csv")
    write_features(list(subs) + list(regression), subs | regression)
    position = write_position(list(subs), subs)
    bal, summ = balance(df, every)
    bal.to_csv(OUT / "balance.csv", index=False)
    summ.to_csv(OUT / "balance_summary.csv", index=False)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip())
    manifest = {"script": "preprocessing/make_subsets.py", "commit": commit, "dirty": dirty,
                "draw": "sklearn train_test_split(cleaned, test_size=size, random_state=seed, stratify=label), test side",
                "random_states": {str(s): list(r) for s, r in SIZES.items()},
                "inputs_sha256": {str(p.relative_to(ROOT)): sha(p) for p in
                                  [RAW] + [q for v in SOURCES.values() for q in v[:2]]},
                "subsets": {t: {"n": len(s), "class1": float(s[LABEL].mean()),
                                "n_qpu_group": int(sum(i in qpu_ids for i in s.index))} for t, s in subs.items()},
                "position": position}
    (OUT / "subsets_manifest.json").write_text(json.dumps(manifest, indent=2))
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 4):
        print(summ.to_string(index=False))
        print("overlap of the subsamples with the actual subsets:")
        print(ov.loc[list(subs), ["CARDS_2500", "CARDS_1000"]].T.to_string())


if __name__ == "__main__":
    main()
