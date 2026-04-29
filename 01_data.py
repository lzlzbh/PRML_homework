from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

root = Path(r"e:\desktop_wj\PRML_HOMEWORK\HOMEWORK3")
csv_path = root / "LSTM-Multivariate_pollution.csv" / "LSTM-Multivariate_pollution.csv"
out_path = root / "minimal_lstm" / "data_lookback18.npz"

lookback = 18
num_cols = ["pollution", "dew", "temp", "press", "wnd_spd", "snow", "rain"]

df = pd.read_csv(csv_path)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce").ffill().bfill()
df["wnd_dir"] = df["wnd_dir"].fillna(df["wnd_dir"].mode().iloc[0])

n = len(df)
n_train = int(n * 0.2)
n_val = int(n * 0.2)
train_df = df.iloc[:n_train].copy()
val_df = df.iloc[n_train:n_train + n_val].copy()
test_df = df.iloc[n_train + n_val:].copy()

train_cat = pd.get_dummies(train_df["wnd_dir"], prefix="wnd_dir")
val_cat = pd.get_dummies(val_df["wnd_dir"], prefix="wnd_dir")
test_cat = pd.get_dummies(test_df["wnd_dir"], prefix="wnd_dir")
cat_cols = sorted(set(train_cat.columns) | set(val_cat.columns) | set(test_cat.columns))
train_cat = train_cat.reindex(columns=cat_cols, fill_value=0)
val_cat = val_cat.reindex(columns=cat_cols, fill_value=0)
test_cat = test_cat.reindex(columns=cat_cols, fill_value=0)

x_scaler = MinMaxScaler()
train_num = x_scaler.fit_transform(train_df[num_cols])
val_num = x_scaler.transform(val_df[num_cols])
test_num = x_scaler.transform(test_df[num_cols])

y_scaler = MinMaxScaler()
y_train = y_scaler.fit_transform(train_df[["pollution"]])
y_val = y_scaler.transform(val_df[["pollution"]])
y_test = y_scaler.transform(test_df[["pollution"]])

train_x = np.hstack([train_num, train_cat.values])
val_x = np.hstack([val_num, val_cat.values])
test_x = np.hstack([test_num, test_cat.values])


def make_windows(x, y, k):
    xs, ys = [], []
    for i in range(k, len(x)):
        xs.append(x[i - k:i])
        ys.append(y[i])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


x_train, y_train = make_windows(train_x, y_train, lookback)
x_val, y_val = make_windows(val_x, y_val, lookback)
x_test, y_test = make_windows(test_x, y_test, lookback)

np.savez_compressed(
    out_path,
    x_train=x_train,
    y_train=y_train,
    x_val=x_val,
    y_val=y_val,
    x_test=x_test,
    y_test=y_test,
)

print("saved:", out_path)
print("x_train:", x_train.shape, "x_val:", x_val.shape, "x_test:", x_test.shape)
