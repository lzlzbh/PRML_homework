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
n=len(x_train)
X=torch.stack([torch.ones(n), x_train], dim=1)
w=torch.zeros(2, requires_grad=True)
pred=X @ w
loss=torch.mean((y_train-pred)**2)
loss.backward()
grad=w.grad
H=2*(X.T @ X)/n
w_new=w-torch.linalg.inv(H) @ grad
w0, w1=w_new[0].item(), w_new[1].item()
y_train_pred=w0+w1*x_train
y_test_pred=w0+w1*x_test
train_rmse, train_r2=evaluate(y_train, y_train_pred)
test_rmse, test_r2=evaluate(y_test, y_test_pred)
print(f"w0={w0:.6f}, w1={w1:.6f}")
print(f"训练集: RMSE={train_rmse:.6f}, R2={train_r2:.6f}")
print(f"测试集: RMSE={test_rmse:.6f}, R2={test_r2:.6f}")