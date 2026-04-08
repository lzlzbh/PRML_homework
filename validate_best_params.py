import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

d = np.load("E:/desktop_wj/PRML_HOMEWORK/HOMEWORK2/CODE/minimal_run/moons3d_data.npz")
X_train, y_train, X_test, y_test = d["X_train"], d["y_train"], d["X_test"], d["y_test"]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

configs = {
    "Decision Tree": (
        DecisionTreeClassifier(random_state=42),
        {"criterion": ["gini", "entropy"], "max_depth": [10, 15, 20, None], "min_samples_leaf": [1, 2, 3, 4, 5]},
    ),
    "AdaBoost + Decision Trees": (
        AdaBoostClassifier(estimator=DecisionTreeClassifier(random_state=42), random_state=42),
        {"estimator__max_depth": [1, 2, 3, 4, 5], "n_estimators": [50, 100, 150, 200, 300], "learning_rate": [0.1, 0.5, 1.0, 1.5, 2.0]},
    ),
    "SVM (Linear Kernel)": (
        Pipeline([("scaler", StandardScaler()), ("svc", SVC(kernel="linear"))]),
        {"svc__C": [0.01, 0.1, 1, 10]},
    ),
    "SVM (RBF Kernel)": (
        Pipeline([("scaler", StandardScaler()), ("svc", SVC(kernel="rbf"))]),
        {"svc__C": [0.1, 0.5, 1, 2, 5], "svc__gamma": [0.01, 0.1, 0.5, 1, "scale"]},
    ),
    "SVM (Polynomial Kernel)": (
        Pipeline([("scaler", StandardScaler()), ("svc", SVC(kernel="poly"))]),
        {"svc__C": [1, 10], "svc__gamma": [0.1, 0.5, 1], "svc__coef0": [0.0, 0.5, 1.0], "svc__degree": [3, 4, 5]},
    ),
}

print("Model\tBest Params\tAccuracy\tPrecision\tRecall\tF1")
for name, (model, param_grid) in configs.items():
    gs = GridSearchCV(model, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
    gs.fit(X_train, y_train)
    pred = gs.best_estimator_.predict(X_test)
    acc = accuracy_score(y_test, pred)
    pre = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    print(f"{name}\t{gs.best_params_}\t{acc:.3f}\t{pre:.3f}\t{rec:.3f}\t{f1:.3f}")
