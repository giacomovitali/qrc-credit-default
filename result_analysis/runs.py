"""Load classification run logs into a table.

    from runs import load_runs
    df = load_runs('results/logs')

One row per (run, resampling method, model) with the five test metrics of the log's metrics line
(`AUC` is the area under the PR curve of the *hard* predictions), the selected
hyperparameters and validation F1. Logs written by run_classification.py also have an
`Extended metrics: {json}` line per model, which adds ROC-AUC and PR-AUC from
continuous scores, balanced accuracy, the confusion matrix, timings and the
parameter count (see EXTENDED). Runs
produced by classification/run_classification.py also carry a manifest, and
its fields are added as columns (`seed`, `split`, `features`, `git_commit`, …),
so figures can group by what a run actually was instead of by substrings of its
filename. Runs without a manifest get those columns empty; their
filename is still available in `run`.

Can also be run as a script to print a summary:

    python result_analysis/runs.py results/logs [--glob 'QRC_*']
"""

import argparse
import ast
import fnmatch
import json
import os
import re
import warnings

import pandas as pd

METRICS = ['Accuracy', 'Precision', 'Recall', 'AUC', 'F1-score']
_METRIC_LINE = re.compile(r'Accuracy: (\S+)\tPrecision: (\S+)\tRecall: (\S+)\tAUC: (\S+)\tF1-score: (\S+)')
_RESAMPLING_LINE = re.compile(r'Classification with aggregated features - (\S+) resampling')
_MODEL_LINE = re.compile(r' - (.+?) classification results\s*$')
_PARAMS_LINE = re.compile(r'Best parameters:\t(.*)$')
_VAL_F1_LINE = re.compile(r'Best validation F1-score:\t(\S+)')
_EXTENDED_LINE = re.compile(r'Extended metrics: (\{.*\})\s*$')
# fields of the extended line, in column order (empty for older logs)
EXTENDED = ['roc_auc', 'pr_auc', 'balanced_acc', 'score_source', 'tn', 'fp', 'fn', 'tp', 'grid_size', 'fit_data',
            'fit_time_s', 'grid_time_s', 'predict_time_s', 'n_train', 'n_test', 'n_params']

# log line -> short model name, as used in the figures
MODEL_NAMES = {
    'KNN': 'KNN', 'SGD': 'SGD', 'Naive Bayes': 'NB', 'Logistic model': 'LM',
    'Perceptron': 'P', 'SVM': 'SVM', 'Linear SVM': 'LSVM', 'Decision tree': 'DT', 'Random Forest': 'RF', 'DNN': 'DNN',
    'Regularised decision tree': 'DTR', 'Regularised random forest': 'RFR', 'Extra trees': 'ET',
    'Gradient boosting': 'HGB',
    'Naive Bayes train only': 'NBT',
}


def parse_log(path):
    """Metric rows of one log. A metrics line with no model line before it is a DNN (older log format)."""
    rows, resampling, model, selection = [], None, None, {}
    with open(path) as f:
        for line in f:
            if m := _RESAMPLING_LINE.search(line):
                resampling, model, selection = m.group(1), None, {}
            elif m := _MODEL_LINE.search(line):
                model, selection = MODEL_NAMES.get(m.group(1), m.group(1)), {}
            elif m := _PARAMS_LINE.search(line):
                selection['best_params'] = m.group(1).strip()
            elif m := _VAL_F1_LINE.search(line):
                selection['val_f1'] = float(m.group(1))
            elif m := _METRIC_LINE.search(line):
                rows.append({'resampling': resampling, 'model': model or 'DNN',
                             **dict(zip(METRICS, map(float, m.groups()))),
                             'best_params': selection.get('best_params'), 'val_f1': selection.get('val_f1'),
                             **dict.fromkeys(EXTENDED)})
                model, selection = None, {}
            elif m := _EXTENDED_LINE.search(line):
                ext = json.loads(m.group(1))
                row = rows[-1]
                if (ext['model'], ext['resampler']) != (row['model'], row['resampling']):
                    raise ValueError(f"{path}: extended metrics for {ext['model']}/{ext['resampler']} "
                                     f"follow the metrics of {row['model']}/{row['resampling']}")
                row.update({k: ext.get(k) for k in EXTENDED})
                row['best_params'] = json.dumps(ext['best_params'])
                row['val_f1'] = ext['val_f1']
    return rows


def load_predictions(path):
    """A <run>.predictions.csv.gz file (keeps the resampler name 'None' as a string)."""
    return pd.read_csv(path, keep_default_na=False, na_values={'score': ['', 'nan', 'NaN']})


def best_params(value):
    """The best_params column as a dict (it holds JSON for new logs, a Python repr for old ones)."""
    if value is None or (isinstance(value, float) and value != value):
        return None
    try:
        return json.loads(value)
    except ValueError:
        return ast.literal_eval(value)


def _manifest_fields(path):
    with open(path) as f:
        m = json.load(f)
    a = m.get('args', {})
    return {
        'status': m.get('status'), 'created': m.get('created'),
        'git_commit': m.get('git_commit'), 'git_dirty': m.get('git_dirty'),
        'split': a.get('split'), 'seed': a.get('seed'), 'split_seed': a.get('split_seed'),
        'resample_seed': a.get('resample_seed'), 'qpu_perc': a.get('qpu_perc') if a.get('split') == 'qpu-mix' else None,
        'features': ' '.join(a.get('features', [])), 'train_features': ' '.join(a.get('train_features') or []),
        'eval_features': ' '.join(a.get('eval_features') or []),
        'alignment': ' '.join(sorted({i['alignment'] for i in m.get('inputs', [])})),
    }


def load_runs(log_dir='results/logs', pattern='*'):
    """All runs in `log_dir` whose name (without .log) matches the glob `pattern`."""
    frames = []
    for name in sorted(os.listdir(log_dir)):
        if not name.endswith('.log') or not fnmatch.fnmatch(name[:-4], pattern):
            continue
        rows = parse_log(os.path.join(log_dir, name))
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df.insert(0, 'run', name[:-4])
        manifest = os.path.join(log_dir, name[:-4] + '.manifest.json')
        df['has_manifest'] = os.path.exists(manifest)
        if df['has_manifest'].iloc[0]:
            for k, v in _manifest_fields(manifest).items():
                df[k] = v
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    with warnings.catch_warnings():  # runs without manifests leave these columns all-NA
        warnings.simplefilter('ignore', FutureWarning)
        return pd.concat(frames, ignore_index=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Summarise classification run logs.')
    ap.add_argument('log_dir', nargs='?', default='results/logs')
    ap.add_argument('--glob', default='*', help="run-name pattern, e.g. 'QRC_*'")
    args = ap.parse_args()
    df = load_runs(args.log_dir, args.glob)
    if df.empty:
        raise SystemExit('No runs found')
    print(f"{df['run'].nunique()} runs, {len(df)} metric rows, {df['has_manifest'].groupby(df['run']).first().sum()} with a manifest")
    print(df.groupby(['model', 'resampling'])['F1-score'].agg(['count', 'mean', 'std']).round(3).to_string())
