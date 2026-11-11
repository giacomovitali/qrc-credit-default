from sklearn.metrics import (accuracy_score, recall_score, precision_score, f1_score, auc, precision_recall_curve,
                             roc_auc_score, average_precision_score, balanced_accuracy_score)
from copy import deepcopy
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from dnn import dnn
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
import json
import logging
import time
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import os

# Directory for confusion-matrix figures; drivers may override it.
FIGURE_DIR = 'results/logs/fig'
# Cores for the random forest (results do not depend on it); drivers may override it.
N_JOBS = -1
# If a list, evaluation_metrics() appends one row per test datapoint (ID, label, prediction, score),
# so test-set statistics (bootstrap, ROC/PR curves) can be computed later without re-running.
PREDICTIONS = None

# Every evaluated model also logs one 'Extended metrics: {json}' line after the legacy metrics line.
# Its wording avoids the substrings the legacy result_analysis/ parsers match on.
EXTENDED_PREFIX = 'Extended metrics: '


def _select_on_validation(grid, make, X_train, Y_train, X_val, Y_val):
    """Fit make(params) for each params in grid, in order; keep the first with the highest
    validation F1 (> 0)."""
    best = {'val_f1': 0, 'params': None, 'model': None, 'fit_time_s': None}
    t0 = time.perf_counter()
    for params in grid:
        model = make(params)
        t = time.perf_counter()
        model.fit(X_train, Y_train)
        fit_time = time.perf_counter() - t
        Y_val_pred = model.predict(X_val)
        f1 = f1_score(Y_val_pred, Y_val)
        if f1 > best['val_f1']:
            best = {'val_f1': f1, 'params': params, 'model': model, 'fit_time_s': fit_time}
    best['grid_time_s'] = time.perf_counter() - t0
    best['grid_size'] = len(grid)
    return best


def _grid_model(name, grid, make, X_train, Y_train, X_val, Y_val, X_test, Y_test, resamp, hyperp_str):
    best = _select_on_validation(grid, make, X_train, Y_train, X_val, Y_val)
    if best['val_f1'] > 0:
        logging.info(f"Best parameters:\t{best['params']}")
        logging.info(f"Best validation F1-score:\t{best['val_f1']}")
        _evaluate(best['model'], X_test, Y_test, name, resamp, hyperp_str, best, n_train=len(Y_train))
    else:
        logging.info('F1-score on the validation set <= 0')


def _evaluate(model, X_test, Y_test, name, resamp, hyperp_str, selection, n_train):
    t = time.perf_counter()
    Y_pred = model.predict(X_test)
    predict_time = time.perf_counter() - t
    scores, score_source = test_scores(model, X_test)
    evaluation_metrics(Y_test, Y_pred, name, resamp, hyperp_str, scores=scores, score_source=score_source,
                       selection=selection, predict_time_s=predict_time, n_train=n_train,
                       n_params=n_parameters(model))


def test_scores(model, X):
    """Continuous class-1 scores for ROC/PR-AUC, and which method produced them."""
    if hasattr(model, 'predict_proba') and not isinstance(model, (SGDClassifier, Perceptron, SVC)):
        return model.predict_proba(X)[:, 1], 'predict_proba'
    return model.decision_function(X), 'decision_function'


def n_parameters(model):
    """Fitted parameters of a readout: weights + biases (linear), means + variances + priors (NB),
    tree nodes (trees and tree ensembles). None for KNN, which stores the training set instead."""
    if isinstance(model, GaussianNB):
        return int(model.theta_.size + model.var_.size + model.class_prior_.size)
    if isinstance(model, DecisionTreeClassifier):
        return int(model.tree_.node_count)
    if isinstance(model, (RandomForestClassifier, ExtraTreesClassifier)):
        return int(sum(t.tree_.node_count for t in model.estimators_))
    if isinstance(model, HistGradientBoostingClassifier):
        return int(sum(len(p.nodes) for it in model._predictors for p in it))
    if hasattr(model, 'coef_'):
        return int(np.size(model.coef_) + np.size(model.intercept_))
    if isinstance(model, torch.nn.Module):
        return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    return None


def NaiveBayes(X_train, Y_train, X_val, Y_val, X_test, Y_test, resamp, hyperp_str=''):
    # As in the paper: no hyperparameters, so it is fitted on train + validation. NaiveBayesTrain
    # is the variant that sees the same training data as every other model.
    model = GaussianNB()
    X_train_f = pd.concat([X_train, X_val], axis=0, ignore_index=True)
    Y_train_f = pd.concat([Y_train, Y_val], axis=0, ignore_index=True)
    t = time.perf_counter()
    model.fit(X_train_f, Y_train_f)
    selection = {'params': {}, 'val_f1': None, 'fit_time_s': time.perf_counter() - t, 'grid_size': 1,
                 'fit_data': 'train+val'}
    selection['grid_time_s'] = selection['fit_time_s']
    _evaluate(model, X_test, Y_test, 'NB', resamp, hyperp_str, selection, n_train=len(Y_train_f))


def NaiveBayesTrain(X_train, Y_train, X_val, Y_val, X_test, Y_test, resamp, hyperp_str=''):
    model = GaussianNB()
    t = time.perf_counter()
    model.fit(X_train, Y_train)
    selection = {'params': {}, 'val_f1': f1_score(model.predict(X_val), Y_val),
                 'fit_time_s': time.perf_counter() - t, 'grid_size': 1, 'fit_data': 'train'}
    selection['grid_time_s'] = selection['fit_time_s']
    _evaluate(model, X_test, Y_test, 'NBT', resamp, hyperp_str, selection, n_train=len(Y_train))


def SGD(X_train, Y_train, X_val, Y_val, X_test, Y_test, resamp, hyperp_str=''):
    grid = [{'n': n} for n in range(10, 200, 10)]
    make = lambda p: SGDClassifier(loss="hinge", penalty="l2", max_iter=p['n'], random_state=42, class_weight='balanced')
    _grid_model('SGD', grid, make, X_train, Y_train, X_val, Y_val, X_test, Y_test, resamp, hyperp_str)


def KNN(X_train, Y_train, X_val, Y_val, X_test, Y_test, resamp, hyperp_str=''):
    X_val = np.ascontiguousarray(X_val)
    X_test = np.ascontiguousarray(X_test)
    grid = [{'n': n} for n in range(2, 16)]
    make = lambda p: KNeighborsClassifier(n_neighbors=p['n'])
    _grid_model('KNN', grid, make, X_train, Y_train, X_val, Y_val, X_test, Y_test, resamp, hyperp_str)


def logistic_model(X_train, Y_train, X_val, Y_val, X_test, Y_test, resamp, hyperp_str=''):
    grid = [{'C': C} for C in [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]]
    make = lambda p: LogisticRegression(random_state=24, class_weight='balanced', max_iter=1000, verbose=False, C=p['C'])
    _grid_model('LM', grid, make, X_train, Y_train, X_val, Y_val, X_test, Y_test, resamp, hyperp_str)


def svm(X_train, Y_train, X_val, Y_val, X_test, Y_test, resamp, hyperp_str=''):
    # The submitted code also looped over gamma in {0.1, 'scale'}, which a linear kernel ignores
    # (identical fits, so the selected model and the test metrics are unchanged without it).
    grid = [{'C': C} for C in [1e-1, 1e0, 1e1, 1e2]]
    make = lambda p: SVC(random_state=24, C=p['C'], kernel='linear', class_weight='balanced')
    _grid_model('SVM', grid, make, X_train, Y_train, X_val, Y_val, X_test, Y_test, resamp, hyperp_str)


def linear_svm(X_train, Y_train, X_val, Y_val, X_test, Y_test, resamp, hyperp_str=''):
    # Linear SVM fitted with liblinear instead of libsvm: same C grid and class weights as svm(),
    # ~50-100x faster on CARDS_30000. Squared hinge loss in the primal: with the plain hinge loss
    # (dual only) liblinear does not converge for C >= 1 on these features, while the primal
    # converges at every C. Unlike SVC it also regularises the intercept.
    grid = [{'C': C} for C in [1e-1, 1e0, 1e1, 1e2]]
    make = lambda p: LinearSVC(C=p['C'], loss='squared_hinge', dual=False, class_weight='balanced', max_iter=20000, random_state=24)
    _grid_model('LSVM', grid, make, X_train, Y_train, X_val, Y_val, X_test, Y_test, resamp, hyperp_str)


def decision_tree(X_train, Y_train, X_val, Y_val, X_test, Y_test, resamp, hyperp_str=''):
    grid = [{'max_depth': md, 'criterium': c} for md in [5, 10, 20, 30, 50] for c in ['entropy', 'log_loss', 'gini']]
    make = lambda p: DecisionTreeClassifier(criterion=p['criterium'], random_state=24, max_depth=p['max_depth'], class_weight='balanced')
    _grid_model('DT', grid, make, X_train, Y_train, X_val, Y_val, X_test, Y_test, resamp, hyperp_str)


def random_forest(X_train, Y_train, X_val, Y_val, X_test, Y_test, resamp, hyperp_str=''):
    grid = [{'n_estimators': n, 'max_features': m} for n in [10, 50, 100, 200] for m in [None, 'sqrt']]
    make = lambda p: RandomForestClassifier(criterion='entropy', random_state=24, class_weight='balanced', n_jobs=N_JOBS,
                                            n_estimators=p['n_estimators'], max_features=p['max_features'])
    _grid_model('RF', grid, make, X_train, Y_train, X_val, Y_val, X_test, Y_test, resamp, hyperp_str)


# Reference baselines (tree models). DT and RF above keep the grids of the submitted code;
# DTR and RFR add leaf-size and depth limits, without which class_weight has little effect on fully
# grown trees. Grids of 15-18 points, comparable to SGD (19) and KNN (14).
def decision_tree_reg(X_train, Y_train, X_val, Y_val, X_test, Y_test, resamp, hyperp_str=''):
    grid = [{'max_depth': md, 'min_samples_leaf': l} for md in [4, 6, 8, 12, None] for l in [1, 20, 100]]
    make = lambda p: DecisionTreeClassifier(random_state=24, class_weight='balanced', **p)
    _grid_model('DTR', grid, make, X_train, Y_train, X_val, Y_val, X_test, Y_test, resamp, hyperp_str)


_FOREST_GRID = [{'max_features': m, 'min_samples_leaf': l, 'max_depth': md}
                for m in ['sqrt', 0.5] for l in [1, 5, 20, 50] for md in [None, 12]]


def random_forest_reg(X_train, Y_train, X_val, Y_val, X_test, Y_test, resamp, hyperp_str=''):
    make = lambda p: RandomForestClassifier(n_estimators=300, random_state=24, class_weight='balanced', n_jobs=N_JOBS, **p)
    _grid_model('RFR', _FOREST_GRID, make, X_train, Y_train, X_val, Y_val, X_test, Y_test, resamp, hyperp_str)


def extra_trees(X_train, Y_train, X_val, Y_val, X_test, Y_test, resamp, hyperp_str=''):
    make = lambda p: ExtraTreesClassifier(n_estimators=300, random_state=24, class_weight='balanced', n_jobs=N_JOBS, **p)
    _grid_model('ET', _FOREST_GRID, make, X_train, Y_train, X_val, Y_val, X_test, Y_test, resamp, hyperp_str)


def hist_gradient_boosting(X_train, Y_train, X_val, Y_val, X_test, Y_test, resamp, hyperp_str=''):
    # early_stopping='auto' would hold out a random 10 % of the training set when it has more than
    # 10,000 rows; disabled so that every model is selected on the same validation set.
    grid = [{'learning_rate': lr, 'max_leaf_nodes': n, 'max_iter': it}
            for lr in [0.03, 0.1, 0.3] for n in [15, 31, 63] for it in [100, 300]]
    make = lambda p: HistGradientBoostingClassifier(early_stopping=False, random_state=24, class_weight='balanced', **p)
    _grid_model('HGB', grid, make, X_train, Y_train, X_val, Y_val, X_test, Y_test, resamp, hyperp_str)


def perceptron(X_train, Y_train, X_val, Y_val, X_test, Y_test, resamp, hyperp_str=''):
    grid = [{'learning_rate': lr} for lr in [1e-1, 1e-2, 1e-3]]
    make = lambda p: Perceptron(eta0=p['learning_rate'], random_state=42, max_iter=1000, class_weight='balanced')
    _grid_model('P', grid, make, X_train, Y_train, X_val, Y_val, X_test, Y_test, resamp, hyperp_str)


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    return str(o)


def evaluation_metrics(Y_test, Y_pred, model, resamp, hyperp_str, scores=None, score_source=None,
                       selection=None, predict_time_s=None, n_train=None, n_params=None):
    Y_pred = np.asarray(Y_pred)
    test_acc = accuracy_score(Y_test, Y_pred)
    test_precision = precision_score(Y_test, Y_pred)
    test_recall = recall_score(Y_test, Y_pred)
    precision, recall, _ = precision_recall_curve(Y_test, Y_pred)
    test_auc = auc(recall, precision)
    test_f1 = f1_score(Y_test, Y_pred)
    # Legacy line, parsed by result_analysis/: 'AUC' is the PR curve of the hard predictions.
    logging.info(f'Accuracy: {test_acc}\tPrecision: {test_precision}\tRecall: {test_recall}\tAUC: {test_auc}\tF1-score: {test_f1}')
    conf_matrix = confusion_matrix(Y_test, Y_pred, labels=[0, 1])
    tn, fp, fn, tp = conf_matrix.ravel()
    selection = selection or {}
    extended = {
        'model': model, 'resampler': resamp,
        'precision': test_precision, 'recall': test_recall, 'f1': test_f1,
        'balanced_acc': balanced_accuracy_score(Y_test, Y_pred),
        'roc_auc': roc_auc_score(Y_test, scores) if scores is not None else None,
        'pr_auc': average_precision_score(Y_test, scores) if scores is not None else None,
        'score_source': score_source,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'best_params': selection.get('params'), 'val_f1': selection.get('val_f1'),
        'grid_size': selection.get('grid_size'), 'fit_data': selection.get('fit_data', 'train'),
        'fit_time_s': selection.get('fit_time_s'), 'grid_time_s': selection.get('grid_time_s'),
        'predict_time_s': predict_time_s, 'n_train': n_train, 'n_test': len(Y_pred), 'n_params': n_params,
    }
    logging.info(EXTENDED_PREFIX + json.dumps(extended, default=_json_default))
    if PREDICTIONS is not None:
        ids = Y_test.index if hasattr(Y_test, 'index') else range(len(Y_pred))
        PREDICTIONS.append(pd.DataFrame({'resampler': resamp, 'model': model, 'ID': ids, 'y_true': np.asarray(Y_test),
                                         'y_pred': Y_pred, 'score': scores if scores is not None else np.nan}))
    # Plot confusion matrix
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'], annot_kws={"fontsize":15})
    plt.savefig(os.path.join(FIGURE_DIR, f'cm_{model}{hyperp_str}_{resamp}.png'))
    plt.close()


def DNN(X_train, Y_train, X_val, Y_val, X_test, Y_test, resamp, hyperp_str=''):
    params = {'learning_rate': [1e-1, 1e-2, 1e-3], 'batch_size': [64, 128, 256]}
    best_f1_score = 0
    t0 = time.perf_counter()
    for lr in params['learning_rate']:
        for bs in params['batch_size']:
            t = time.perf_counter()
            f1, model = dnn(X_train, Y_train, X_val, Y_val, bs, lr)
            fit_time = time.perf_counter() - t
            if f1 > best_f1_score:
                best_f1_score = f1
                best_params = {'learning_rate': lr, 'batch_size': bs}
                best_model = deepcopy(model)
                best_fit_time = fit_time
    grid_time = time.perf_counter() - t0
    if best_f1_score>0:
        logging.info(f"Best parameters:\t{best_params}")
        logging.info(f"Best validation F1-score:\t{best_f1_score}")
        X_test_t = torch.from_numpy(X_test.values.astype(np.float32))
        t = time.perf_counter()
        with torch.no_grad():
            outputs = best_model(X_test_t)
        _, Y_pred = torch.max(outputs, 1)
        predict_time = time.perf_counter() - t
        scores = torch.softmax(outputs, 1)[:, 1].numpy()
        selection = {'params': best_params, 'val_f1': best_f1_score, 'fit_time_s': best_fit_time,
                     'grid_time_s': grid_time, 'grid_size': 9}
        evaluation_metrics(Y_test, Y_pred.numpy(), 'DNN', resamp, hyperp_str, scores=scores, score_source='softmax',
                           selection=selection, predict_time_s=predict_time, n_train=len(Y_train),
                           n_params=n_parameters(best_model))
    else:
        logging.info('F1-score on the validation set <= 0')
