import torch
import torch
import pandas as pd
train_df=pd.read_csv(r"E:\desktop_wj\PRML_HOMEWORK\HOMEWORK1\DATA\Data4Regression - Training Data.csv")
test_df=pd.read_csv(r"E:\desktop_wj\PRML_HOMEWORK\HOMEWORK1\DATA\Data4Regression - Test Data.csv")
x_train_col='x_new' if 'x_new' in train_df.columns else 'x'
y_train_col='y_new_complex' if 'y_new_complex' in train_df.columns else 'y_complex'
x_test_col='x_new' if 'x_new' in test_df.columns else 'x'
y_test_col='y_new_complex' if 'y_new_complex' in test_df.columns else 'y_complex'
x_train=torch.tensor(train_df[x_train_col].values, dtype=torch.float32)
y_train=torch.tensor(train_df[y_train_col].values, dtype=torch.float32)
x_test=torch.tensor(test_df[x_test_col].values, dtype=torch.float32)
y_test=torch.tensor(test_df[y_test_col].values, dtype=torch.float32)
def evaluate(truth, pred):
    rmse=torch.sqrt(torch.mean((truth-pred)**2)).item()
    ss_res=torch.sum((truth-pred)**2)
    ss_tot=torch.sum((truth-torch.mean(truth))**2)
    r2=(1-ss_res/ss_tot).item()
    return rmse, r2
bandwidths=[0.05, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
print(f"{'bandwidth':<12} {'train_rmse':<12} {'train_r2':<12} {'test_rmse':<12} {'test_r2':<12}")
print("-"*60)
for bw in bandwidths:
    dist_train=x_train[:, None]-x_train[None, :]
    W_train=torch.exp(-0.5*(dist_train/bw)**2)
    W_train=W_train/W_train.sum(dim=1, keepdim=True)
    y_train_pred=W_train @ y_train
    dist_test=x_test[:, None]-x_train[None, :]
    W_test=torch.exp(-0.5*(dist_test/bw)**2)
    W_test=W_test/W_test.sum(dim=1, keepdim=True)
    y_test_pred=W_test @ y_train
    train_rmse, train_r2=evaluate(y_train, y_train_pred)
    test_rmse, test_r2=evaluate(y_test, y_test_pred)
    print(f"{bw:<12} {train_rmse:<12.6f} {train_r2:<12.6f} {test_rmse:<12.6f} {test_r2:<12.6f}")