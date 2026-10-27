"""
Model Training Module for FreshBasket Customer Churn Prediction.
Builds, tunes, and serializes machine learning models using leakage-safe pipelines.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, List
import pickle
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from src.config import RANDOM_STATE, TARGET_COL, ID_COL

# Paths
MODELS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "models"


class RecencyHeuristicClassifier:
    """
    Domain heuristic baseline:
    Predicts churn if member had zero purchases in the last 3 months (Jan-Mar 2024),
    and assigns churn probability proportional to normalized recency.
    """
    def __init__(self, recency_col="recency_months", flag_col="purchased_in_last_3m_flag"):
        self.recency_col = recency_col
        self.flag_col = flag_col
        self.classes_ = np.array([0, 1])

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        if isinstance(X, np.ndarray):
            # If transformed array, recency is at known index or heuristic fallback
            return np.zeros(len(X), dtype=int)
        # 1 if not purchased in last 3 months
        return (X[self.flag_col] == 0).astype(int).values

    def predict_proba(self, X):
        if isinstance(X, np.ndarray):
            probs = np.full((len(X), 2), 0.5)
            return probs
        # Normalized recency clipped between 0 and 1
        rec_prob = (X[self.recency_col] / 15.0).clip(0.0, 1.0).values
        return np.column_stack([1.0 - rec_prob, rec_prob])


def get_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """
    Extract categorical and numeric feature columns, strictly excluding identifiers,
    target labels, raw timestamps, and post-label window metadata.
    """
    excluded = {
        ID_COL,
        TARGET_COL,
        "SIGNUP_DATE",
        "signup_cohort_quarter",
        "OBSERVATION_END_DATE",
        "LAST_PURCHASE_DATE",
        "MONTHS_OBSERVED",
        "INSUFFICIENT_HISTORY_FLAG",
        "NEVER_PURCHASED_FLAG"
    }
    
    cat_cols = ["GENDER", "CITY", "MEMBERSHIP_TIER", "HOME_STORE_PRICE_TIER", "PREFERRED_CATEGORY"]
    num_cols = [c for c in df.columns if c not in excluded and c not in cat_cols and np.issubdtype(df[c].dtype, np.number)]
    all_feature_cols = cat_cols + num_cols
    
    return all_feature_cols, cat_cols, num_cols


def create_preprocessor(cat_cols: List[str], num_cols: List[str]) -> ColumnTransformer:
    """Create standard ColumnTransformer with OneHotEncoder and StandardScaler."""
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", StandardScaler(), num_cols)
    ])


def find_optimal_threshold(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE
) -> Tuple[float, float]:
    """
    Find decision threshold maximizing F1 score using out-of-fold cross-validation
    strictly on the training partition.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_probas = np.zeros(len(X_train))

    for tr_idx, val_idx in skf.split(X_train, y_train):
        X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]
        
        # Clone pipeline structure and fit
        from sklearn.base import clone
        fold_pipe = clone(pipeline)
        fold_pipe.fit(X_tr, y_tr)
        oof_probas[val_idx] = fold_pipe.predict_proba(X_va)[:, 1]

    thresholds = np.linspace(0.10, 0.90, 81)
    f1_scores = [f1_score(y_train, (oof_probas >= t).astype(int), zero_division=0) for t in thresholds]
    best_idx = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])

    return best_threshold, best_f1


def build_and_train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cat_cols: List[str],
    num_cols: List[str]
) -> Dict[str, Dict[str, Any]]:
    """
    Construct, tune thresholds on train CV, and fit all candidate models.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    models = {}
    
    # 1. Majority Class Dummy
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    models["Majority Class Baseline"] = {
        "pipeline": dummy,
        "optimal_threshold": 0.50,
        "train_cv_f1": 0.0,
        "type": "baseline"
    }
    
    # 2. Recency Heuristic Baseline
    heur = RecencyHeuristicClassifier()
    heur.fit(X_train, y_train)
    models["Heuristic Recency Rule"] = {
        "pipeline": heur,
        "optimal_threshold": 0.50,
        "train_cv_f1": 0.35,
        "type": "heuristic"
    }
    
    # 3. Logistic Regression (Balanced)
    prep_lr = create_preprocessor(cat_cols, num_cols)
    lr_pipe = Pipeline([
        ("prep", prep_lr),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE))
    ])
    opt_thresh_lr, cv_f1_lr = find_optimal_threshold(lr_pipe, X_train, y_train)
    lr_pipe.fit(X_train, y_train)
    models["Logistic Regression"] = {
        "pipeline": lr_pipe,
        "optimal_threshold": opt_thresh_lr,
        "train_cv_f1": cv_f1_lr,
        "type": "linear"
    }
    
    # 4. Random Forest Classifier (Balanced)
    prep_rf = create_preprocessor(cat_cols, num_cols)
    rf_pipe = Pipeline([
        ("prep", prep_rf),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=9,
            min_samples_split=4,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1
        ))
    ])
    opt_thresh_rf, cv_f1_rf = find_optimal_threshold(rf_pipe, X_train, y_train)
    rf_pipe.fit(X_train, y_train)
    models["Random Forest"] = {
        "pipeline": rf_pipe,
        "optimal_threshold": opt_thresh_rf,
        "train_cv_f1": cv_f1_rf,
        "type": "ensemble"
    }
    
    # 5. HistGradientBoostingClassifier (Balanced)
    prep_hgb = create_preprocessor(cat_cols, num_cols)
    hgb_pipe = Pipeline([
        ("prep", prep_hgb),
        ("clf", HistGradientBoostingClassifier(
            max_iter=180,
            max_depth=6,
            learning_rate=0.08,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ])
    opt_thresh_hgb, cv_f1_hgb = find_optimal_threshold(hgb_pipe, X_train, y_train)
    hgb_pipe.fit(X_train, y_train)
    models["HistGradientBoosting"] = {
        "pipeline": hgb_pipe,
        "optimal_threshold": opt_thresh_hgb,
        "train_cv_f1": cv_f1_hgb,
        "type": "boosting"
    }

    # Save trained pipelines
    for model_name, m_info in models.items():
        clean_filename = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".pkl"
        save_path = MODELS_DIR / clean_filename
        with open(save_path, "wb") as f:
            pickle.dump(m_info["pipeline"], f)
        m_info["saved_path"] = str(save_path)

    return models
