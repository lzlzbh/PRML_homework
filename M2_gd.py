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
#算评价指标
n=len(x_train)
X=torch.stack([torch.ones(n), x_train], dim=1)#构造矩阵
lr=1e-3
epochs=20000
w=torch.zeros(2, requires_grad=True)
for i in range(epochs):#前向传播
    pred=X @ w
    loss=torch.mean((y_train-pred)**2)
    loss.backward()
    with torch.no_grad():
        w -= lr*w.grad
        w.grad.zero_()#梯度清零
w0, w1=w[0].item(), w[1].item()
y_train_pred=w0+w1*x_train
y_test_pred=w0+w1*x_test
train_rmse, train_r2=evaluate(y_train, y_train_pred)
test_rmse, test_r2=evaluate(y_test, y_test_pred)
print(f"w0={w0:.6f}, w1={w1:.6f}")
print(f"训练集: RMSE={train_rmse:.6f}, R2={train_r2:.6f}")
print(f"测试集: RMSE={test_rmse:.6f}, R2={test_r2:.6f}")
