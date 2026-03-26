import torch
import pandas as pd
train_df=pd.read_csv(r"E:\desktop_wj\PRML_HOMEWORK\HOMEWORK1\DATA\Data4Regression - Training Data.csv")
test_df=pd.read_csv(r"E:\desktop_wj\PRML_HOMEWORK\HOMEWORK1\DATA\Data4Regression - Test Data.csv")
x_train = torch.tensor(train_df['x'].values, dtype=torch.float32)
y_train = torch.tensor(train_df['y_complex'].values, dtype=torch.float32)
x_test = torch.tensor(test_df['x_new'].values, dtype=torch.float32)
y_test = torch.tensor(test_df['y_new_complex'].values, dtype=torch.float32)
#从csv取出来数据，并整成pytorch张量
def evaluate(truth, pred):
    rmse=torch.sqrt(torch.mean((truth-pred)**2)).item()
    ss_res=torch.sum((truth-pred)**2)
    ss_tot=torch.sum((truth-torch.mean(truth))**2)
    r2=(1-ss_res/ss_tot).item()
    return rmse, r2
max_degree=20
print(f"{'degree':<8} {'train_rmse':<12} {'train_r2':<12} {'test_rmse':<12} {'test_r2':<12}")
print("-"*56)
for d in range(1, max_degree+1):
    X_train=torch.stack([x_train**i for i in range(d+1)], dim=1)
    X_test=torch.stack([x_test**i for i in range(d+1)], dim=1)
    w=torch.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
    y_train_pred=X_train @ w
    y_test_pred=X_test @ w
    train_rmse, train_r2=evaluate(y_train, y_train_pred)
    test_rmse, test_r2=evaluate(y_test, y_test_pred)
    print(f"{d:<8} {train_rmse:<12.6f} {train_r2:<12.6f} {test_rmse:<12.6f} {test_r2:<12.6f}")
