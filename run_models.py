import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


data = np.load("E:/desktop_wj/PRML_HOMEWORK/HOMEWORK2/CODE/minimal_run/moons3d_data.npz")//npz_data
X_train, y_train = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]

models = {
    "Decision Tree": DecisionTreeClassifier(
        criterion="entropy", max_depth=None, min_samples_leaf=3, random_state=42
    ),
    "AdaBoost + Decision Trees": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=5, random_state=42),
        n_estimators=200,
        learning_rate=1.5,
        random_state=42,
    ),
    "SVM (Linear Kernel)": Pipeline(
        [("scaler", StandardScaler()), ("svc", SVC(kernel="linear", C=0.1))]
    ),
    "SVM (RBF Kernel)": Pipeline(
        [("scaler", StandardScaler()), ("svc", SVC(kernel="rbf", C=2, gamma=1))]
    ),
    "SVM (Polynomial Kernel)": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="poly", C=10, gamma=0.5, coef0=1.0, degree=4)),
        ]
    ),
}//所需的四个模型

print("Model\tAccuracy\tPrecision\tRecall\tF1")
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    pre = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    print(f"{name}\t{acc:.3f}\t{pre:.3f}\t{rec:.3f}\t{f1:.3f}")//输出结果
