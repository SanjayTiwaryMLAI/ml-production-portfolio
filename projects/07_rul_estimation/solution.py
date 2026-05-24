"""
Use Case: Remaining Useful Life (RUL) Estimation
-------------------------------------------------
Problem : Estimate hours until equipment failure for maintenance planning.
Approach: RUL label engineering → Degradation features → LightGBM regression.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error


def generate_degradation_data(n_assets=40, seed=42):
    rng, frames = np.random.default_rng(seed), []
    for aid in range(n_assets):
        life  = rng.integers(500, 2000)
        t     = np.arange(life)
        deg   = t / life
        df    = pd.DataFrame({
            "asset_id":         aid,
            "time_step":        t,
            "transformer_temp": 70 + 20*deg + rng.normal(0,3,life),
            "oil_temperature":  60 + 15*deg + rng.normal(0,2,life),
            "output_power_kw":  100 - 30*deg + rng.normal(0,5,life),
            "vibration":        1 + 4*deg**2 + rng.normal(0,0.3,life),
            "fault_count":      (rng.poisson(0.05,life)*(1+5*deg)).astype(int),
            "rul":              life - t - 1,
        })
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def add_features(df):
    df = df.sort_values(["asset_id","time_step"]).copy()
    for col in ["transformer_temp","oil_temperature","output_power_kw","vibration"]:
        df[f"{col}_r10_mean"] = df.groupby("asset_id")[col].transform(lambda x: x.rolling(10,min_periods=1).mean())
        df[f"{col}_delta"]    = df.groupby("asset_id")[col].transform(lambda x: x.diff().fillna(0))
    df["fault_cumsum"]    = df.groupby("asset_id")["fault_count"].cumsum()
    df["normalized_time"] = df.groupby("asset_id")["time_step"].transform(lambda x: x/x.max())
    df["temp_ratio"]      = df["transformer_temp"] / (df["oil_temperature"]+1e-8)
    return df.dropna()


def train_rul(df):
    df    = add_features(df)
    excl  = ["asset_id","time_step","rul"]
    X, y, g = df[[c for c in df.columns if c not in excl]], df["rul"], df["asset_id"]
    tr, te  = next(GroupShuffleSplit(1, test_size=0.2, random_state=42).split(X,y,g))
    model   = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.04, max_depth=7,
                                 random_state=42, verbose=-1)
    model.fit(X.iloc[tr], y.iloc[tr])
    mae = mean_absolute_error(y.iloc[te], model.predict(X.iloc[te]))
    print(f"MAE: {mae:.1f} hours ({mae/24:.1f} days)")
    return model, [c for c in df.columns if c not in excl]


if __name__ == "__main__":
    df = generate_degradation_data(40)
    print(f"Dataset: {len(df):,} records | {df['asset_id'].nunique()} assets")
    model, features = train_rul(df)

    df_feat = add_features(df)
    latest  = df_feat.sort_values("time_step").groupby("asset_id").tail(1).copy()
    latest["pred_rul_hours"] = np.maximum(0, model.predict(latest[features]))
    latest["pred_rul_days"]  = (latest["pred_rul_hours"]/24).round(1)
    latest["risk"] = pd.cut(latest["pred_rul_days"], bins=[-1,3,7,30,np.inf],
                            labels=["CRITICAL","HIGH","MEDIUM","LOW"])
    print("\nAsset RUL Predictions:")
    print(latest[["asset_id","pred_rul_hours","pred_rul_days","risk"]].sort_values("pred_rul_hours").head(10).to_string(index=False))
