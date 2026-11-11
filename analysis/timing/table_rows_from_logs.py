"""Rows of the timing table measured on 100 repeated runs of one CARDS_30000 split.

The KNN, SGD, Naive Bayes, Logistic Model and Perceptron rows (QRC features, detuning encoding) and the
DNN row of the timing table come from repeated runs of an earlier version of the classification driver,
whose logs carry a timestamp on every line (deposit: results/timing/repeated_runs/). The libsvm SVC row
is computed as well, for reference; the table's random-forest and linear-SVM rows come from timing.py.

  QRC_detuning_full_dataset_240offset_timing_<i>.log   KNN SGD NB LM P
  QRC_detuning_full_dataset_<i>.log                    SVM (libsvm SVC, linear kernel)
  DNN_full_dataset_timing_<i>.log                      DNN

Classifiers: per run and resampler, the time from the "<model> classification results" line to the
following metrics line (the whole hyperparameter search, test evaluation included) divided by the number
of grid points, i.e. the time per configuration; mean and population SD over the runs.
Grid points: KNN 14, SGD 19 (epochs 10-190), NB 1, LM 6, P 3, SVM 4.
DNN: per run and resampler, the time from the first epoch line to early stopping (one training).

    python analysis/timing/table_rows_from_logs.py [--logs results/timing/repeated_runs] [--out results/timing]

Writes <out>/repeated_runs_rows.csv. Run from the repository root (conda env qrc_class).
"""
import argparse
import glob
import os
import re
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

RESAMPLERS = ["None", "SMOTE", "kSMOTE", "CC"]
MODELS = {"KNN": ("KNN", 14), "SGD": ("SGD", 19), "Naive Bayes": ("NB", 1), "Logistic model": ("LM", 6),
          "Perceptron": ("P", 3), "SVM": ("SVM", 4)}
STAMP = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")
HEADER = re.compile(r"Classification with aggregated features - (.+?) resampling")


def stamp(line):
    m = STAMP.match(line)
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f") if m else None


def classifier_times(path, wanted):
    """{(model, resampler): [seconds]} of one log: search start to the next metrics line."""
    times = defaultdict(list)
    lines = open(path).readlines()
    resampler = None
    for i, line in enumerate(lines):
        m = HEADER.search(line)
        if m:
            resampler = m.group(1).strip()
            continue
        for key in wanted:
            if f"{key} classification results" in line and (t0 := stamp(line)):
                for nxt in lines[i + 1:]:
                    if "Classification with aggregated features -" in nxt:
                        break
                    if all(f"{k}: " in nxt for k in ("Accuracy", "AUC", "Recall", "Precision", "F1-score")):
                        if (t1 := stamp(nxt)):
                            times[(key, resampler)].append((t1 - t0).total_seconds())
                        break
    return times


def dnn_times(path):
    """{resampler: [seconds]} of one DNN log: first epoch to early stopping."""
    times = defaultdict(list)
    lines = open(path).readlines()
    i = 0
    while i < len(lines):
        m = HEADER.search(lines[i])
        if m:
            j = i + 1
            while j < len(lines) and not re.search(r"Epoch \d+/", lines[j]):
                j += 1
            if j >= len(lines):
                break
            t0 = datetime.strptime(lines[j].split(" - ")[0], "%Y-%m-%d %H:%M:%S,%f")
            while j < len(lines) and "Early stopping" not in lines[j]:
                j += 1
            if j >= len(lines):
                break
            t1 = datetime.strptime(lines[j].split(" - ")[0], "%Y-%m-%d %H:%M:%S,%f")
            times[m.group(1).strip()].append((t1 - t0).total_seconds())
            i = j
        i += 1
    return times


def collect(pattern, wanted, logs):
    out = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(logs, pattern))):
        for k, v in classifier_times(path, wanted).items():
            out[k].extend(v)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--logs", default="results/timing/repeated_runs")
    ap.add_argument("--out", default="results/timing")
    args = ap.parse_args()

    readouts = ["KNN", "SGD", "Naive Bayes", "Logistic model", "Perceptron"]
    times = collect("QRC_detuning_full_dataset_240offset_timing_*.log", readouts, args.logs)
    svm_logs = [p for p in glob.glob(os.path.join(args.logs, "QRC_detuning_full_dataset_*.log")) if "240" not in os.path.basename(p)]
    for path in svm_logs:
        for k, v in classifier_times(path, ["SVM"]).items():
            times[k].extend(v)
    dnn = defaultdict(list)
    for path in glob.glob(os.path.join(args.logs, "DNN_full_dataset_timing_*.log")):
        for k, v in dnn_times(path).items():
            dnn[k].extend(v)

    rows = []
    for key, (name, grid) in MODELS.items():
        for r in RESAMPLERS:
            t = np.array(times.get((key, r), []))
            if len(t):
                rows.append(dict(model=name, resampler=r, mean_s=t.mean() / grid, sd_s=t.std() / grid, n_runs=len(t), grid_points=grid))
    for r in RESAMPLERS:
        t = np.array(dnn.get(r, []))
        if len(t):
            rows.append(dict(model="DNN", resampler=r, mean_s=t.mean(), sd_s=t.std(), n_runs=len(t), grid_points=1))
    df = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, "repeated_runs_rows.csv"), index=False)
    table = df.assign(cell=[f"{m:.2f} ± {s:.2f}" for m, s in zip(df.mean_s, df.sd_s)]).pivot(index="model", columns="resampler", values="cell")
    print(table.reindex(columns=RESAMPLERS).to_string())


if __name__ == "__main__":
    main()
