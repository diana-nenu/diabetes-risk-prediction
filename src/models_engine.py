from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """
    Trains and evaluates multiple models using class balancing techniques.
    """
    # 1. Calculate the ratio for XGBoost to handle class imbalance
    num_healthy = (y_train == 0).sum()
    num_diabetes = (y_train == 1).sum()
    ratio = num_healthy / num_diabetes if num_diabetes > 0 else 1

    # 2. Define models using 'balanced' weights for LR and RF
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            scale_pos_weight=ratio,
            eval_metric='logloss',
            random_state=42
        )
    }

    results = {}

    for name, model in models.items():
        # Training
        model.fit(X_train, y_train)

        # Prediction
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Metric Collection
        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "AUC": roc_auc_score(y_test, y_proba)
        }

    return pd.DataFrame(results).T, models


def get_feature_importance(model, feature_names):
    """
    Extracts and sorts feature importance.
    Works for RF, XGBoost, and Logistic Regression (using coefficients).
    """
    # For Tree-based models (RF, XGBoost)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    # For Linear models (Logistic Regression) - using absolute coefficient values
    elif hasattr(model, 'coef_'):
        importances = model.coef_[0]
    else:
        return "The selected model does not support direct feature importance extraction."

    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    return feature_importance_df