"""
Use Case: Elasticity-Driven Pricing Optimisation
--------------------------------------------------
Problem : Replace flat markup rules with elasticity-aware pricing that
          maximises revenue while maintaining seller adoption rates.
Approach: Seller cohort segmentation → LightGBM demand model (monotonic
          constraints) → Constrained revenue optimisation → Weekly pricing table.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import minimize_scalar
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error


def generate_shipment_data(n=20000, seed=42):
    rng = np.random.default_rng(seed)
    df  = pd.DataFrame({
        "cohort":         rng.choice(["high_volume", "mid_volume", "low_volume"], n),
        "route":          rng.choice(["US-IN", "US-UK", "US-DE", "US-AU"], n),
        "markup_pct":     rng.uniform(5, 40, n),
        "weight_kg":      rng.uniform(0.1, 30, n),
        "peak_season":    rng.integers(0, 2, n),
        "competitor_idx": rng.uniform(0.8, 1.2, n),
    })
    base = {"high_volume": 500, "mid_volume": 150, "low_volume": 30}
    elst = {"high_volume": -1.5, "mid_volume": -2.0, "low_volume": -2.8}
    df["base"]  = df["cohort"].map(base)
    df["elast"] = df["cohort"].map(elst)
    df["shipment_volume"] = np.maximum(
        0, df["base"] * (1 + df["elast"] * df["markup_pct"] / 100 + rng.normal(0, 0.1, n))
    ).astype(int)
    return df.drop(columns=["base", "elast"])


def train_demand_model(df):
    df_feat  = pd.get_dummies(df, columns=["cohort", "route"], drop_first=True)
    features = [c for c in df_feat.columns if c != "shipment_volume"]
    X, y     = df_feat[features], df_feat["shipment_volume"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    # markup_pct must decrease demand: monotone constraint = -1
    constraints = [-1] + [0] * (len(features) - 1)
    model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, max_depth=6,
                               monotone_constraints=constraints,
                               monotone_constraints_method="advanced",
                               random_state=42, verbose=-1)
    model.fit(Xtr, ytr)
    mape = mean_absolute_percentage_error(yte, model.predict(Xte))
    print(f"  Demand Model MAPE: {mape:.2%}")
    return model, features


def optimise_markup(model, features, base_row, markup_range=(5,40), min_adoption=0.7):
    low_row  = {**base_row, "markup_pct": markup_range[0]}
    base_vol = model.predict(pd.DataFrame([low_row]))[0]
    min_vol  = base_vol * min_adoption

    def neg_rev(m):
        vol = model.predict(pd.DataFrame([{**base_row, "markup_pct": m}]))[0]
        return -(m * vol) if vol >= min_vol else 0

    res     = minimize_scalar(neg_rev, bounds=markup_range, method="bounded")
    opt_m   = round(res.x, 2)
    opt_vol = model.predict(pd.DataFrame([{**base_row, "markup_pct": opt_m}]))[0]
    return {"optimal_markup_pct": opt_m, "expected_volume": round(opt_vol,1),
            "expected_revenue": round(opt_m * opt_vol, 2)}


if __name__ == "__main__":
    print("Training demand model...")
    df    = generate_shipment_data()
    model, features = train_demand_model(df)

    # Create a sample base row with correct feature columns
    base = {f: 0 for f in features}
    base.update({"markup_pct": 10, "weight_kg": 5, "peak_season": 0, "competitor_idx": 1.0})
    if "cohort_mid_volume" in base: base["cohort_mid_volume"] = 1
    if "route_US-UK" in base:       base["route_US-UK"] = 1

    result = optimise_markup(model, features, base)
    print(f"\nOptimal markup: {result['optimal_markup_pct']}%")
    print(f"Expected volume: {result['expected_volume']} shipments")
    print(f"Expected revenue: ${result['expected_revenue']:,.2f}")
