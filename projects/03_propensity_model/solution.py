"""
Use Case: Program Adoption Propensity Model
--------------------------------------------
Problem : Identify sellers most likely to adopt a warehousing and distribution
          program to replace inefficient heuristic-based campaign targeting.
Approach: Feature engineering from behavioural/transactional data →
          LightGBM binary classifier → Precision@K evaluation → A/B validation.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.utils import resample


def generate_seller_data(n=10000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "tenure_months":             rng.integers(1, 60, n),
        "monthly_shipment_volume":   rng.integers(10, 5000, n),
        "self_ship_rate":            rng.uniform(0, 1, n),
        "return_rate":               rng.uniform(0.01, 0.25, n),
        "active_listings":           rng.integers(1, 500, n),
        "listing_update_freq":       rng.uniform(0.1, 10.0, n),
        "seller_central_logins_30d": rng.integers(0, 40, n),
        "avg_order_value":           rng.uniform(50, 2000, n),
        "categories_count":          rng.integers(1, 10, n),
    })
    score = (0.3 * (df["monthly_shipment_volume"] / 5000) +
             0.2 * df["self_ship_rate"] +
             0.15 * (df["tenure_months"] / 60) +
             0.15 * (df["seller_central_logins_30d"] / 40) +
             0.1  * rng.uniform(0, 1, n))
    df["adopted"] = (score > 0.5).astype(int)
    return df


def engineer_features(df):
    df = df.copy()
    df["revenue_proxy"]       = df["monthly_shipment_volume"] * df["avg_order_value"]
    df["engagement_score"]    = df["seller_central_logins_30d"] * df["listing_update_freq"]
    df["high_volume_flag"]    = (df["monthly_shipment_volume"] > 200).astype(int)
    df["experienced_seller"]  = (df["tenure_months"] > 12).astype(int)
    return df


def precision_at_k(y_true, y_scores, k):
    top_k = np.argsort(y_scores)[::-1][:k]
    return y_true[top_k].mean()


def train_propensity_model(df):
    df = engineer_features(df)
    X, y = df.drop(columns=["adopted"]), df["adopted"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Balance classes
    data = pd.concat([X_train, y_train], axis=1)
    maj  = data[data["adopted"]==0]
    mi   = resample(data[data["adopted"]==1], n_samples=len(maj), random_state=42)
    bal  = pd.concat([maj, mi])
    X_bal, y_bal = bal.drop(columns=["adopted"]), bal["adopted"]

    model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                class_weight="balanced", random_state=42, verbose=-1)
    model.fit(X_bal, y_bal)
    y_scores = model.predict_proba(X_test)[:, 1]

    return {
        "model":              model,
        "auc_roc":            round(roc_auc_score(y_test, y_scores), 4),
        "avg_precision":      round(average_precision_score(y_test, y_scores), 4),
        "precision_at_500":   round(precision_at_k(y_test.values, y_scores, 500), 4),
        "precision_at_1000":  round(precision_at_k(y_test.values, y_scores, 1000), 4),
        "top_features":       dict(pd.Series(model.feature_importances_,
                                             index=X.columns).nlargest(5)),
    }


if __name__ == "__main__":
    print("Generating seller data...")
    df = generate_seller_data(10000)
    print(f"Dataset: {len(df)} sellers | Adoption rate: {df['adopted'].mean():.1%}")
    print("\nTraining propensity model...")
    r = train_propensity_model(df)
    print(f"AUC-ROC: {r['auc_roc']} | Avg Precision: {r['avg_precision']}")
    print(f"Precision@500: {r['precision_at_500']} | Precision@1000: {r['precision_at_1000']}")
    print("Top Features:", r["top_features"])
