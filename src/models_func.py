from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score


def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf


def train_xgboost(X_train, y_train):
    xgb = XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, random_state=42, n_jobs=-1,
        scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1])
    )
    xgb.fit(X_train, y_train)
    return xgb


def train_lightgbm(X_train, y_train, best_params=None):
    if best_params is None:
        best_params = {'colsample_bytree': 0.8, 'learning_rate': 0.05,
                       'max_depth': 6, 'n_estimators': 500, 'num_leaves': 63, 'subsample': 0.8}
    lgb = LGBMClassifier(**best_params, random_state=42, n_jobs=-1)
    lgb.fit(X_train, y_train)
    return lgb


def evaluate_model(model, X_test, y_test):
    """predict, and return metrics and predictions"""

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_proba)

    return y_pred, y_proba, report, roc_auc
