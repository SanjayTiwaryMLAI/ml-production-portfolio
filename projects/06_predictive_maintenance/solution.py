"""
Use Case: Predictive Maintenance — Equipment Failure Prediction
----------------------------------------------------------------
Problem : Predict inverter/transformer failures 24h-7 days in advance.
Approach: Label engineering (pre-failure windows) → Feature engineering →
          LightGBM classifier → Asset-level risk scoring.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score


def generate_telemetry(n_assets=30, n_days=180, seed=42):
    rng, frames = np.random.default_rng(seed), []
    for asset in range(n_assets):
        n  = n_days * 24
        ts = pd.date_range("2023-01-01", periods=n, freq="h")
        df = pd.DataFrame({
            "asset_id":        asset,
            "timestamp":       ts,
            "temperature_c":   65 + rng.normal(0, 5, n),
            "oil_temp_c":      55 + rng.normal(0, 4, n),
            "output_power_kw": 100 + rng.normal(0, 10, n),
            "vibration_mm_s":  2 + rng.uniform(0, 1, n),
            "fault_codes":     rng.poisson(0.1, n),
            "failure":         0,
        })
        for ft in sorted(rng.choice(np.arange(72, n-24), size=rng.integers(2,5), replace=False)):
            df.loc[ft, "failure"] = 1
            w = min(48, ft)
            df.loc[ft-w:ft, "temperature_c"]  += np.linspace(0, 20, w+1)[:w+1]
            df.loc[ft-w:ft, "vibration_mm_s"] += np.linspace(0, 5, w+1)[:w+1]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def add_labels(df, horizon_h=168):
    df = df.sort_values(["asset_id","timestamp"]).copy()
    df["label"] = 0
    for aid, grp in df.groupby("asset_id"):
        for ft in grp[grp["failure"]==1]["timestamp"].values:
            mask = ((df["asset_id"]==aid) &
                    (df["timestamp"] >= pd.Timestamp(ft) - pd.Timedelta(hours=horizon_h)) &
                    (df["timestamp"] < pd.Timestamp(ft)))
            df.loc[mask, "label"] = 1
    return df


def add_features(df):
    df = df.sort_values(["asset_id","timestamp"]).copy()
    for col in ["temperature_c","oil_temp_c","output_power_kw","vibration_mm_s"]:
        df[f"{col}_r6_mean"]  = df.groupby("asset_id")[col].transform(lambda x: x.rolling(6, min_periods=1).mean())
        df[f"{col}_r24_std"]  = df.groupby("asset_id")[col].transform(lambda x: x.rolling(24,min_periods=1).std().fillna(0))
    df["fault_cumsum_7d"] = df.groupby("asset_id")["fault_codes"].transform(lambda x: x.rolling(168,min_periods=1).sum())
    df["temp_delta"]      = df["temperature_c"] - df["oil_temp_c"]
    return df.dropna()


def train_model(df):
    df  = add_features(add_labels(df))
    exc = ["asset_id","timestamp","failure","label"]
    X, y, g = df[[c for c in df.columns if c not in exc]], df["label"], df["asset_id"]
    tr, te  = next(GroupShuffleSplit(1, test_size=0.2, random_state=42).split(X,y,g))
    model   = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, scale_pos_weight=5,
                                  random_state=42, verbose=-1)
    model.fit(X.iloc[tr], y.iloc[tr])
    auc = roc_auc_score(y.iloc[te], model.predict_proba(X.iloc[te])[:,1])
    print(f"AUC-ROC: {auc:.4f}")
    return model, [c for c in df.columns if c not in exc]


if __name__ == "__main__":
    print("Generating telemetry...")
    df = generate_telemetry(30, 180)
    print(f"{len(df):,} records | {df['asset_id'].nunique()} assets")
    print("Training model...")
    model, features = train_model(df)

    # Score assets
    df_feat = add_features(add_labels(df))
    latest  = df_feat.sort_values("timestamp").groupby("asset_id").tail(1).copy()
    latest["risk_score"] = model.predict_proba(latest[features])[:,1]
    print("\nTop 5 High-Risk Assets:")
    print(latest[["asset_id","risk_score"]].sort_values("risk_score",ascending=False).head().to_string(index=False))
