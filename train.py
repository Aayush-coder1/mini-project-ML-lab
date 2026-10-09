"""End-to-end training: Employee Turnover Prediction 2026.
Models (as per reference project, modernised): Decision Tree + Random Forest.
- stratified 80/20 split, class_weight='balanced' for imbalance
- GridSearchCV tuning, 5-fold CV
- metrics: accuracy / precision / recall / F1 / ROC-AUC + confusion matrix
- saves: models/*.pkl, models/metrics.json, models/feature_names.json
Run: python train.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

BASE = Path(__file__).parent
DATA = BASE / "data" / "employee_attrition_hr_2026.csv"
OUT = BASE / "models"
OUT.mkdir(exist_ok=True)

TARGET = "Attrition"
DROP = ["EmployeeID"]

CATEGORICAL = ["Gender", "MaritalStatus", "Department", "JobRole",
               "WorkMode", "BusinessTravel", "AIJobRisk"]
NUMERIC = ["Age", "JobLevel", "MonthlyIncome", "StockOptionLevel", "PercentSalaryHike",
           "TotalWorkingYears", "YearsAtCompany", "YearsInCurrentRole",
           "YearsSinceLastPromotion", "NumCompaniesWorked", "CommuteMinutes",
           "OvertimeHours", "TrainingHoursLastYear", "ManagerSupport", "BurnoutScore",
           "EngagementScore", "JobSatisfaction", "WorkLifeBalance", "PerformanceRating",
           "AIToolsUsageHours"]


def load():
    df = pd.read_csv(DATA)
    print(f"Dataset: {df.shape} | Attrition Yes={(df[TARGET] == 'Yes').mean():.3f}")
    print(f"Missing values: {int(df.isna().sum().sum())}")
    y = (df[TARGET] == "Yes").astype(int)
    X = df.drop(columns=[TARGET] + DROP)
    return X, y


def make_pre():
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", "passthrough", NUMERIC),
    ])


def evaluate(name, model, X_test, y_test):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    m = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }
    print(f"\n== {name} ==")
    for k, v in m.items():
        if k != "confusion_matrix":
            print(f"  {k:10s}: {v}")
    print(f"  confusion: {m['confusion_matrix']}")
    print(classification_report(y_test, pred, target_names=["Stayed", "Left"], zero_division=0))
    return m


def main():
    X, y = load()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    print(f"Train {X_train.shape} (pos={y_train.mean():.3f}) | Test {X_test.shape} (pos={y_test.mean():.3f})")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ---- Decision Tree ----
    dt_pipe = Pipeline([("pre", make_pre()),
                        ("clf", DecisionTreeClassifier(random_state=42, class_weight="balanced"))])
    dt_grid = {"clf__max_depth": [4, 6, 8, 10, None],
               "clf__min_samples_split": [2, 5, 10],
               "clf__criterion": ["gini", "entropy"]}
    dt = GridSearchCV(dt_pipe, dt_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
    dt.fit(X_train, y_train)
    print(f"Best DT params: {dt.best_params_} (cv AUC={dt.best_score_:.4f})")

    # ---- Random Forest ----
    rf_pipe = Pipeline([("pre", make_pre()),
                        ("clf", RandomForestClassifier(random_state=42, class_weight="balanced_subsample", n_jobs=-1))])
    rf_grid = {"clf__n_estimators": [200, 300],
               "clf__max_depth": [8, 12, None],
               "clf__min_samples_split": [2, 5]}
    rf = GridSearchCV(rf_pipe, rf_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
    rf.fit(X_train, y_train)
    print(f"Best RF params: {rf.best_params_} (cv AUC={rf.best_score_:.4f})")

    m_dt = evaluate("Decision Tree (tuned)", dt.best_estimator_, X_test, y_test)
    m_rf = evaluate("Random Forest (tuned)", rf.best_estimator_, X_test, y_test)

    # ---- feature importances (from RF) ----
    pre = rf.best_estimator_.named_steps["pre"]
    feat_names = list(pre.get_feature_names_out())
    importances = rf.best_estimator_.named_steps["clf"].feature_importances_
    order = np.argsort(importances)[::-1][:15]
    top = [{"feature": feat_names[i].split("__")[-1] if "__" in feat_names[i] else feat_names[i],
            "importance": round(float(importances[i]), 4)} for i in order]
    print("\nTop 15 features (RF):")
    for t in top:
        print(f"  {t['feature']:35s} {t['importance']}")

    joblib.dump(dt.best_estimator_, OUT / "attrition_dt.pkl")
    joblib.dump(rf.best_estimator_, OUT / "attrition_rf.pkl")
    with open(OUT / "metrics.json", "w") as f:
        json.dump({"decision_tree": m_dt, "random_forest": m_rf,
                   "best_params": {"dt": dt.best_params_, "rf": rf.best_params_},
                   "top_features": top,
                   "test_size": len(y_test),
                   "positive_rate": round(float(y.mean()), 4)}, f, indent=2)
    with open(OUT / "feature_info.json", "w") as f:
        json.dump({"categorical": CATEGORICAL, "numeric": NUMERIC,
                   "target": TARGET, "drop": DROP}, f, indent=2)
    print(f"\nSaved models + metrics -> {OUT}/")


if __name__ == "__main__":
    main()
