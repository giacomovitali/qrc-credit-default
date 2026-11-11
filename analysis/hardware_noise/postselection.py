"""Post-selection statistics of the Aquila runs.

For every hardware task (one datapoint at one readout time): shots returned, register copies,
copies kept by the filling filter (every trap of the copy filled in the pre-sequence, as in
`bitstrings_filtered`), and the effective records per readout. Kept counts are checked against
the filtered records the features were reduced from. Reads the raw records of export_raw.py
(results/hardware_noise/raw/).

    python analysis/hardware_noise/postselection.py
"""
import numpy as np
import pandas as pd

from common import OUT, Records

NSITES = {"detuning": 12, "position": 13}
RAW = OUT / "raw"


def main():
    rows, site_rows = [], []
    for enc, prefix in (("detuning", "aquila_det"), ("position", "aquila_pos")):
        k = NSITES[enc]
        counts = Records(prefix).counts()
        shots = np.load(RAW / f"{prefix}_shots.npz")
        pre_all, completed = shots["pre"], shots["completed"]
        for t in pd.read_csv(RAW / f"{prefix}_shots_index.csv").itertuples():
            ok = completed[t.offset:t.offset + t.n_shots]
            if not ok.any():
                print(f"{enc} {t.run} datapoint {t.id} readout {t.readout}: no completed shot ({t.n_shots} returned)")
                continue
            pre = pre_all[t.offset:t.offset + t.n_shots][ok]
            copies = pre.shape[1] // k
            filled = pre.reshape(len(pre), copies, k).all(axis=2)
            vac = 1 - pre.reshape(len(pre), copies, k).mean(axis=(0, 1))
            requested = None if pd.isna(t.requested_shots) else int(t.requested_shots)
            rows.append(dict(encoding=enc, run=t.run, id=t.id, readout=t.readout, requested_shots=requested,
                             shots=t.n_shots, completed_shots=int(ok.sum()), copies=copies,
                             records_before=int(ok.sum()) * copies, records_kept=int(filled.sum()),
                             kept_stored=int(counts.loc[t.id, t.readout])))
            site_rows.append(dict(encoding=enc, run=t.run, id=t.id, readout=t.readout,
                                  **{f"vacancy_site{i + 1}": v for i, v in enumerate(vac)}))
    df = pd.DataFrame(rows)
    df["kept_fraction"] = df.records_kept / df.records_before
    mismatch = df[df.records_kept != df.kept_stored]
    print(f"{len(df)} tasks; kept counts differing from the stored filtered records: {len(mismatch)}")
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "postselection_tasks.csv", index=False)
    pd.DataFrame(site_rows).to_csv(OUT / "postselection_site_vacancy.csv", index=False)

    q = lambda s: pd.Series({"median": s.median(), "q25": s.quantile(.25), "q75": s.quantile(.75),
                             "min": s.min(), "max": s.max(), "mean": s.mean()})
    summ = []
    for (enc, run), g in df.groupby(["encoding", "run"]):
        summ.append(dict(encoding=enc, run=run, datapoints=g.id.nunique(), tasks=len(g),
                         shots_per_task=int(g.shots.median()), copies=int(g.copies.iloc[0]),
                         records_before=int(g.records_before.median()),
                         **{f"kept_frac_{k}": v for k, v in q(g.kept_fraction).items()},
                         **{f"records_kept_{k}": v for k, v in q(g.records_kept).items()}))
    for enc, g in df.groupby("encoding"):
        summ.append(dict(encoding=enc, run="all", datapoints=g.id.nunique(), tasks=len(g),
                         **{f"kept_frac_{k}": v for k, v in q(g.kept_fraction).items()},
                         **{f"records_kept_{k}": v for k, v in q(g.records_kept).items()}))
    summ = pd.DataFrame(summ)
    summ.to_csv(OUT / "postselection_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.precision", 3):
        print(summ)


if __name__ == "__main__":
    main()
