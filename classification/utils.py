from sklearn.model_selection import train_test_split
import pandas as pd


def create_dataset(X_emu, X_qpu, Y_emu, Y_qpu, qpu_perc, test_size, val_size, random_state=7):
    if qpu_perc == 0:
        X_qpu_cs = X_qpu
        Y_qpu_cs = Y_qpu
        X_qpu_ts = pd.DataFrame()
        Y_qpu_ts = pd.DataFrame()
    elif qpu_perc == 1:
        X_qpu_ts = X_qpu
        Y_qpu_ts = Y_qpu
        X_qpu_cs = pd.DataFrame()
        Y_qpu_cs = pd.DataFrame()
    else:
        X_qpu_cs, X_qpu_ts, Y_qpu_cs, Y_qpu_ts = train_test_split(X_qpu, Y_qpu, test_size=qpu_perc, random_state=random_state, stratify=Y_qpu)
    emu_ts_samples = test_size - len(Y_qpu_ts)
    X_emu_cs, X_emu_ts, Y_emu_cs, Y_emu_ts = train_test_split(X_emu, Y_emu, test_size=emu_ts_samples, random_state=random_state, stratify=Y_emu)
    X_test = pd.concat([X_qpu_ts, X_emu_ts])
    Y_test = pd.concat([Y_qpu_ts, Y_emu_ts])
    X_cs = pd.concat([X_qpu_cs, X_emu_cs])
    Y_cs = pd.concat([Y_qpu_cs, Y_emu_cs])
    X_train, X_val, Y_train, Y_val = train_test_split(X_cs, Y_cs, test_size=val_size, random_state=random_state, stratify=Y_cs)
    return X_train, X_val, X_test, Y_train, Y_val, Y_test