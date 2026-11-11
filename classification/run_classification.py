"""Train and evaluate classifiers on reservoir (or raw) features.

Replaces the three legacy drivers (classification_small_data.py,
classification_full_data.py, classification_small_heter.py) with one
argument-driven script. Run from the repository root, in the `qrc_class` env.

Examples
--------
Random stratified 70/15/15 split over the QPU and emulation groups together,
as in the small-dataset runs of the paper:

    python classification/run_classification.py \\
        --features datasets/position_qpu_sv.csv datasets/position_emu_sv.csv \\
        --labels dataset_aquila/labels_qpu.csv dataset_aquila/labels_emu.csv \\
        --classifiers SVM --seed 1 --run-name QRC_position_small_dataset_sv_1

Test set drawn partly from QPU-processed samples, as in the full-dataset runs
(pairs are given as emulation group first, QPU group second):

    python classification/run_classification.py --split qpu-mix --qpu-perc 0.4 \\
        --features dataset_aquila/emulation_samples_240offset.csv dataset_aquila/qpu_samples_240offset.csv \\
        --labels dataset_aquila/labels_emu.csv dataset_aquila/labels_qpu.csv \\
        --seed 7 --resample-seed 24 --run-name QRC_full_dataset_0.4perc

Train on one backend, evaluate on another (same datapoints, different features):

    python classification/run_classification.py \\
        --features datasets/position_qpu_aquila.csv --labels dataset_aquila/labels_qpu.csv \\
        --train-features datasets/position_qpu_sv.csv --seed 1 --run-name QRC_small_heter_sv_train_1

Outputs
-------
<log-dir>/<run-name>.log            metrics, in the format result_analysis/ parses
<log-dir>/<run-name>.manifest.json  arguments, git commit, package versions,
                                    input checksums, split sizes, resamplers used
<log-dir>/<run-name>.predictions.csv.gz  per test datapoint: label, prediction
                                    and score of every model (--no-predictions
                                    to skip)
<figure-dir>/cm_<model>_<run-name>_<resampling>.png

The run name names the outputs; the manifest records what the run was in structured form.
"""

import argparse
import datetime
import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import classifiers
from resampling_methods import resample_dataset, describe_resampler
from utils import create_dataset

# name on the command line -> (function, line logged before it runs)
CLASSIFIERS = {
    'KNN': (classifiers.KNN, 'KNN classification results'),
    'SGD': (classifiers.SGD, 'SGD classification results'),
    'NB': (classifiers.NaiveBayes, 'Naive Bayes classification results'),
    'NBT': (classifiers.NaiveBayesTrain, 'Naive Bayes train only classification results'),
    'LM': (classifiers.logistic_model, 'Logistic model classification results'),
    'P': (classifiers.perceptron, 'Perceptron classification results'),
    'SVM': (classifiers.svm, 'SVM classification results'),
    'LSVM': (classifiers.linear_svm, 'Linear SVM classification results'),
    'DT': (classifiers.decision_tree, 'Decision tree classification results'),
    'RF': (classifiers.random_forest, 'Random Forest classification results'),
    'DTR': (classifiers.decision_tree_reg, 'Regularised decision tree classification results'),
    'RFR': (classifiers.random_forest_reg, 'Regularised random forest classification results'),
    'ET': (classifiers.extra_trees, 'Extra trees classification results'),
    'HGB': (classifiers.hist_gradient_boosting, 'Gradient boosting classification results'),
    'DNN': (classifiers.DNN, 'DNN classification results'),
}
# SVM (libsvm SVC, linear kernel) is available but not run by default: on CARDS_30000
# it costs hours per fit. LSVM is the same linear hinge-loss model fitted with liblinear.
# NB is fitted on train + validation; NBT, the default, on train only like every
# other model. DTR, RFR, ET and HGB are the reference baselines (tree models), run on request.
DEFAULT_CLASSIFIERS = ['KNN', 'SGD', 'NBT', 'LM', 'P', 'LSVM', 'DT', 'RF']
RESAMPLING = ['None', 'SMOTE', 'kSMOTE', 'CC']


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_labels(path):
    labels = pd.read_csv(path, index_col='ID').default_payment_next_month
    labels.index = labels.index.astype(int)
    return labels


def load_features(path, label_ids, layout='auto'):
    """Return (X, alignment): X has one row per label ID, in label order.

    Two on-disk layouts exist. Reservoir outputs have one *column* per
    datapoint, named by datapoint ID; preprocessed feature files have one *row*
    per datapoint and an `ID` column. Rows are matched to labels by ID. Files
    whose columns carry no IDs (e.g. the `*offset.csv` merges, whose columns are
    all named `x1`) can only be matched by position, which is correct only if
    their column order is the label order; that case is reported.
    """
    df = pd.read_csv(path)
    df = df.loc[:, [c for c in df.columns if not str(c).startswith('Unnamed')]]
    if layout == 'auto':
        layout = 'rows' if 'ID' in df.columns else 'columns'

    wanted = [int(i) for i in label_ids]
    if layout == 'rows':
        X = df.set_index('ID')
        X.index = X.index.astype(int)
        missing = sorted(set(wanted) - set(X.index))
        if missing:
            raise ValueError(f"{path}: {len(missing)} labelled IDs have no features, e.g. {missing[:5]}")
        X, alignment = X.loc[wanted], 'id'
    else:
        columns = [str(c) for c in df.columns]
        if all(c.isdigit() for c in columns):
            by_id = dict(zip((int(c) for c in columns), df.columns))
            missing = sorted(set(wanted) - set(by_id))
            if missing:
                raise ValueError(f"{path}: {len(missing)} labelled IDs have no features, e.g. {missing[:5]}")
            X, alignment = df[[by_id[i] for i in wanted]].T, 'id'
        else:
            if df.shape[1] != len(wanted):
                raise ValueError(f"{path}: {df.shape[1]} columns without IDs cannot be matched to {len(wanted)} labels")
            warnings.warn(f"{path}: columns carry no datapoint IDs; matching them to labels by position")
            X, alignment = df.T, 'positional'
    X.index = wanted
    X.columns = [str(c) for c in range(X.shape[1])]
    return X.astype(float), alignment


def load_pairs(feature_paths, label_paths, layout):
    if len(feature_paths) != len(label_paths):
        raise SystemExit(f"--features ({len(feature_paths)}) and --labels ({len(label_paths)}) must pair up")
    pairs, provenance = [], []
    for fp, lp in zip(feature_paths, label_paths):
        Y = load_labels(lp)
        X, alignment = load_features(fp, Y.index, layout)
        pairs.append((X, Y))
        provenance.append({'features': fp, 'labels': lp, 'alignment': alignment,
                           'n_datapoints': len(Y), 'n_features': X.shape[1],
                           'sha256': {fp: sha256(fp), lp: sha256(lp)}})
    return pairs, provenance


def override_features(X, paths, layout):
    """Replace X's features, datapoint by datapoint, with those from other files."""
    frames = []
    for p in paths:
        df = pd.read_csv(p, nrows=0)
        cols = [c for c in df.columns if not str(c).startswith('Unnamed')]
        if 'ID' in cols:
            ids = pd.read_csv(p, usecols=['ID']).ID.astype(int)
        else:
            ids = pd.Index([int(c) for c in cols if str(c).isdigit()])
        present = [i for i in X.index if i in set(ids)]
        if present:
            frames.append(load_features(p, present, layout)[0])
    if not frames:
        raise SystemExit(f"None of the override files {paths} contain the datapoints to replace")
    Z = pd.concat(frames)
    missing = sorted(set(X.index) - set(Z.index))
    if missing:
        raise SystemExit(f"Override files lack {len(missing)} datapoints, e.g. {missing[:5]}")
    Z = Z.loc[X.index]
    if Z.shape[1] != X.shape[1]:
        raise SystemExit(f"Override features have {Z.shape[1]} columns, expected {X.shape[1]}")
    return Z


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def git_state():
    def run(*cmd):
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    try:
        return {'git_commit': run('git', 'rev-parse', 'HEAD'),
                'git_dirty': bool(run('git', 'status', '--porcelain', '--untracked-files=no'))}
    except (OSError, subprocess.CalledProcessError):
        return {'git_commit': None, 'git_dirty': None}


def package_versions():
    versions = {}
    for name in ('numpy', 'pandas', 'sklearn', 'imblearn', 'torch'):
        try:
            versions[name] = __import__(name).__version__
        except ImportError:
            versions[name] = None
    return versions


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--features', nargs='+', required=True, help='feature file(s), paired with --labels')
    ap.add_argument('--labels', nargs='+', required=True, help='label file(s), with ID and default_payment_next_month')
    ap.add_argument('--layout', choices=['auto', 'columns', 'rows'], default='auto',
                    help='one column per datapoint (reservoir output) or one row (preprocessed); default: detect')
    ap.add_argument('--split', choices=['random', 'qpu-mix'], default='random',
                    help='random: stratified 70/15/15 over all pairs. qpu-mix: two pairs (emulation, QPU); '
                         'the test set holds --test-size samples, a --qpu-perc fraction of the QPU group')
    ap.add_argument('--qpu-perc', type=float, default=0.0, help='qpu-mix: fraction of the QPU group put in the test set')
    ap.add_argument('--test-size', type=int, default=4441, help='qpu-mix: test-set size')
    ap.add_argument('--val-size', type=int, default=4440, help='qpu-mix: validation-set size')
    ap.add_argument('--train-features', nargs='+', help='take training-set features from these files instead')
    ap.add_argument('--eval-features', nargs='+', help='take validation- and test-set features from these files instead')
    ap.add_argument('--classifiers', nargs='+', choices=list(CLASSIFIERS), default=DEFAULT_CLASSIFIERS)
    ap.add_argument('--resampling', nargs='+', choices=RESAMPLING, default=RESAMPLING)
    ap.add_argument('--seed', type=int, default=7, help='seed for the split, the resampler and model initialisation')
    ap.add_argument('--split-seed', type=int, help='seed for the split only (default: --seed)')
    ap.add_argument('--resample-seed', type=int, help='seed for the resampler only (default: --seed)')
    ap.add_argument('--run-name', required=True, help='names the log, manifest and figures')
    ap.add_argument('--log-dir', default='results/logs')
    ap.add_argument('--figure-dir', default='results/logs/fig')
    ap.add_argument('--n-jobs', type=int, default=-1, help='cores for the random forest (default: all)')
    ap.add_argument('--no-predictions', action='store_true', help='do not write the per-datapoint test predictions')
    args = ap.parse_args(argv)
    args.split_seed = args.seed if args.split_seed is None else args.split_seed
    args.resample_seed = args.seed if args.resample_seed is None else args.resample_seed
    return args


def make_split(args, pairs):
    if args.split == 'random':
        X = pd.concat([x for x, _ in pairs])
        Y = pd.concat([y for _, y in pairs])
        if X.index.duplicated().any():
            raise SystemExit('A datapoint ID appears in more than one input pair')
        X_train, X_rest, Y_train, Y_rest = train_test_split(X, Y, test_size=0.3, random_state=args.split_seed, stratify=Y)
        X_val, X_test, Y_val, Y_test = train_test_split(X_rest, Y_rest, test_size=0.5, random_state=args.split_seed, stratify=Y_rest)
    else:
        if len(pairs) != 2:
            raise SystemExit('--split qpu-mix takes exactly two pairs: emulation group first, QPU group second')
        (X_emu, Y_emu), (X_qpu, Y_qpu) = pairs
        X_train, X_val, X_test, Y_train, Y_val, Y_test = create_dataset(
            X_emu, X_qpu, Y_emu, Y_qpu, args.qpu_perc, args.test_size, args.val_size, random_state=args.split_seed)
    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def main(argv=None):
    args = parse_args(argv)
    for d in (args.log_dir, args.figure_dir):
        os.makedirs(d, exist_ok=True)
    classifiers.FIGURE_DIR = args.figure_dir
    classifiers.N_JOBS = args.n_jobs
    classifiers.PREDICTIONS = None if args.no_predictions else []
    log_path = os.path.join(args.log_dir, f'{args.run_name}.log')
    manifest_path = os.path.join(args.log_dir, f'{args.run_name}.manifest.json')
    predictions_path = os.path.join(args.log_dir, f'{args.run_name}.predictions.csv.gz')
    logging.basicConfig(filename=log_path, filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)
    logging.captureWarnings(True)

    pairs, provenance = load_pairs(args.features, args.labels, args.layout)
    X_train, X_val, X_test, Y_train, Y_val, Y_test = make_split(args, pairs)
    if args.train_features:
        X_train = override_features(X_train, args.train_features, args.layout)
    if args.eval_features:
        X_val = override_features(X_val, args.eval_features, args.layout)
        X_test = override_features(X_test, args.eval_features, args.layout)

    manifest = {
        'created': datetime.datetime.now().isoformat(timespec='seconds'),
        'command': [os.path.relpath(sys.argv[0])] + sys.argv[1:],
        'args': vars(args),
        **git_state(),
        'python_version': platform.python_version(),
        'packages': package_versions(),
        'inputs': provenance,
        'overrides': {p: sha256(p) for p in (args.train_features or []) + (args.eval_features or [])},
        'split_sizes': {'train': len(Y_train), 'val': len(Y_val), 'test': len(Y_test)},
        'positive_fraction': {'train': float(Y_train.mean()), 'val': float(Y_val.mean()), 'test': float(Y_test.mean())},
        'resamplers': {},
        'status': 'running',
    }
    if manifest['git_dirty']:
        warnings.warn('Working tree has uncommitted changes; git_commit does not fully describe this run')

    def write_manifest():
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

    write_manifest()
    logging.info('Run seeds: ' + json.dumps({k: getattr(args, k) for k in ('seed', 'split_seed', 'resample_seed')}))
    t0 = time.time()
    hyperp_str = f'_{args.run_name}'
    for resampling in args.resampling:
        logging.info(f'\nClassification with aggregated features - {resampling} resampling\n')
        Xr, Yr = X_train, Y_train
        if resampling != 'None':
            t = time.perf_counter()
            Xr, Yr, sampler = resample_dataset(X_train, Y_train, resampling, random_seed=args.resample_seed, return_sampler=True)
            if Xr is not None:
                used = {**describe_resampler(sampler), 'time_s': round(time.perf_counter() - t, 3),
                        'n_before': [int((Y_train == c).sum()) for c in (0, 1)],
                        'n_after': [int((Yr == c).sum()) for c in (0, 1)]}
                manifest['resamplers'][resampling] = used
                logging.info('Resampler used: ' + json.dumps(used))
        if Xr is None:
            logging.info('Resampling method exception')
            continue
        for name in args.classifiers:
            fn, message = CLASSIFIERS[name]
            logging.info(message)
            seed_everything(args.seed)
            fn(Xr, Yr, X_val, Y_val, X_test, Y_test, resampling, hyperp_str=hyperp_str)

    if classifiers.PREDICTIONS:
        pd.concat(classifiers.PREDICTIONS, ignore_index=True).to_csv(predictions_path, index=False)
        manifest['predictions'] = os.path.basename(predictions_path)
    manifest['status'] = 'completed'
    manifest['wall_time_s'] = round(time.time() - t0, 1)
    write_manifest()
    print(f'{args.run_name}: log {log_path}, manifest {manifest_path}')


if __name__ == '__main__':
    main()
