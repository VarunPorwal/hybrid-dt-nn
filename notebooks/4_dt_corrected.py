# =============================================================================
# Seoul Bike Sharing Demand — Hybrid DT + NN Model
# Corrected & Production-Ready Version
# Run this ENTIRE script in Google Colab to train and save all artifacts
# =============================================================================

# ── STEP 0: Imports ──────────────────────────────────────────────────────────
import os
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import optuna
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm

# Suppress Optuna logs for cleaner output
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("✅ Imports OK")
print(f"   TensorFlow version : {tf.__version__}")
print(f"   Optuna version     : {optuna.__version__}")


# ── STEP 1: Load Data ────────────────────────────────────────────────────────
data = pd.read_csv("SeoulBikeData.csv", encoding='latin1')

print(f"\n✅ Data loaded — shape: {data.shape}")
print(f"   Columns: {list(data.columns)}")
print(f"   Nulls  : {data.isnull().sum().sum()} total missing values")


# ── STEP 2: Date Feature Engineering ─────────────────────────────────────────
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)

data['Day']     = data['Date'].dt.day
data['Month']   = data['Date'].dt.month
data['Year']    = data['Date'].dt.year
data['Weekday'] = data['Date'].dt.weekday   # 0=Monday, 6=Sunday

data.drop('Date', axis=1, inplace=True)

print(f"\n✅ Date features extracted — new shape: {data.shape}")


# ── STEP 3: Encode Categorical Columns ───────────────────────────────────────
# FIX: Use a SEPARATE LabelEncoder for each column.
# The original code reused one encoder (le.fit_transform 3 times),
# which means only the last column's mapping was preserved. This corrupts
# production inference because you can never decode Seasons or Holiday.

le_seasons    = LabelEncoder()
le_holiday    = LabelEncoder()
le_functioning = LabelEncoder()

data['Seasons']         = le_seasons.fit_transform(data['Seasons'])
data['Holiday']         = le_holiday.fit_transform(data['Holiday'])
data['Functioning Day'] = le_functioning.fit_transform(data['Functioning Day'])

# Bundle encoders for saving
encoders = {
    "Seasons":         le_seasons,
    "Holiday":         le_holiday,
    "Functioning Day": le_functioning
}

print("\n✅ Categorical encoding complete")
print(f"   Seasons classes    : {list(le_seasons.classes_)}")
print(f"   Holiday classes    : {list(le_holiday.classes_)}")
print(f"   Functioning Day classes: {list(le_functioning.classes_)}")


# ── STEP 4: Train / Test Split ────────────────────────────────────────────────
TARGET = 'Rented Bike Count'

X = data.drop(TARGET, axis=1)
y = data[TARGET]

FEATURE_COLUMNS = list(X.columns)   # preserve exact column order!

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n✅ Train/test split done")
print(f"   X_train : {X_train.shape}")
print(f"   X_test  : {X_test.shape}")
print(f"   Feature columns ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")


# ── STEP 5: Scale Features ────────────────────────────────────────────────────
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("\n✅ Scaling complete")
print(f"   Scaler mean (first 3): {scaler.mean_[:3].round(4)}")


# ── STEP 6: Decision Tree ─────────────────────────────────────────────────────
dt = DecisionTreeRegressor(max_depth=5, random_state=42)
dt.fit(X_train_scaled, y_train)

y_pred_train = dt.predict(X_train_scaled)
y_pred_test  = dt.predict(X_test_scaled)

n_train, p = X_train_scaled.shape
n_test     = X_test_scaled.shape[0]

r2_train     = r2_score(y_train, y_pred_train)
r2_test      = r2_score(y_test, y_pred_test)
adj_r2_train = 1 - (1 - r2_train) * (n_train - 1) / (n_train - p - 1)
adj_r2_test  = 1 - (1 - r2_test)  * (n_test  - 1) / (n_test  - p - 1)

print("\n✅ Decision Tree trained")
print(f"   R²          (Train) : {r2_train:.4f}")
print(f"   R²          (Test)  : {r2_test:.4f}")
print(f"   Adjusted R² (Train) : {adj_r2_train:.4f}")
print(f"   Adjusted R² (Test)  : {adj_r2_test:.4f}")


# ── STEP 7: Leaf Mapping ──────────────────────────────────────────────────────
leaf_ids_train = dt.apply(X_train_scaled)

leaf_to_points = {}
for idx, leaf in enumerate(leaf_ids_train):
    leaf_to_points.setdefault(leaf, []).append(idx)

print(f"\n✅ Total DT leaves    : {dt.get_n_leaves()}")
print(f"   Leaves with data  : {len(leaf_to_points)}")


# ── STEP 8: Keras NN Builder ──────────────────────────────────────────────────
def build_model(optimizer="adam", activation="relu", hidden_layers=1, units=64):
    """Build a Keras sequential regression model — matches training architecture."""
    model = keras.Sequential()
    for _ in range(hidden_layers):
        if activation == "leakyrelu":
            model.add(layers.Dense(units))
            model.add(layers.LeakyReLU(negative_slope=0.1))
        else:
            model.add(layers.Dense(units, activation=activation))
    model.add(layers.Dense(1))  # single output for regression
    model.compile(optimizer=optimizer, loss="mse")
    return model


# ── STEP 9: Per-Leaf NN Training with Optuna HPO ─────────────────────────────
def train_nn_leaf(X_leaf, y_leaf, n_trials=30):
    """Train a Keras NN on a single DT leaf using Optuna HPO."""

    def objective(trial):
        model = build_model(
            optimizer=trial.suggest_categorical(
                "optimizer", ["adam", "adamw", "rmsprop", "nadam"]
            ),
            activation=trial.suggest_categorical(
                "activation", ["relu", "leakyrelu", "elu", "tanh", "swish"]
            ),
            hidden_layers=trial.suggest_int("hidden_layers", 1, 5),
            units=trial.suggest_categorical("units", [8, 16, 32, 64, 128])
        )
        model.fit(X_leaf, y_leaf, epochs=30, batch_size=32, verbose=0)
        y_pred = model.predict(X_leaf, verbose=0).flatten()
        return r2_score(y_leaf, y_pred)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params

    # Re-train final model with best params for more epochs
    final_model = build_model(
        optimizer=best_params["optimizer"],
        activation=best_params["activation"],
        hidden_layers=best_params["hidden_layers"],
        units=best_params["units"]
    )
    final_model.fit(X_leaf, y_leaf, epochs=50, batch_size=32, verbose=0)

    return final_model, best_params


# ── STEP 10: Train One NN Per Qualifying Leaf ─────────────────────────────────
MIN_SAMPLES = 50

leaf_models  = {}   # {leaf_id: keras.Model}
leaf_metrics = {}   # {leaf_id: {r2, adj_r2, n_samples, best_params}}

unique_leaves = np.unique(leaf_ids_train)
print(f"\n✅ Starting per-leaf NN training")
print(f"   Total leaves to check : {len(unique_leaves)}")

for leaf_id in unique_leaves:
    idx    = np.where(leaf_ids_train == leaf_id)[0]
    n_leaf = len(idx)

    if n_leaf < MIN_SAMPLES:
        print(f"   Leaf {leaf_id:4d} → SKIPPED ({n_leaf} samples < {MIN_SAMPLES})")
        continue

    X_leaf = X_train_scaled[idx]
    y_leaf = y_train.iloc[idx].values

    print(f"   Leaf {leaf_id:4d} → Training NN ({n_leaf} samples) ...", end=" ", flush=True)
    model, best_params = train_nn_leaf(X_leaf, y_leaf)

    y_leaf_pred  = model.predict(X_leaf, verbose=0).flatten()
    r2_leaf      = r2_score(y_leaf, y_leaf_pred)
    adj_r2_leaf  = 1 - (1 - r2_leaf) * (n_leaf - 1) / (n_leaf - p - 1)

    leaf_models[leaf_id] = model
    leaf_metrics[leaf_id] = {
        "num_samples": int(n_leaf),
        "r2":          round(float(r2_leaf), 4),
        "adjusted_r2": round(float(adj_r2_leaf), 4),
        "best_params": best_params
    }
    print(f"R²={r2_leaf:.4f}")

print(f"\n✅ Leaf NNs trained: {len(leaf_models)} / {len(unique_leaves)} leaves")


# ── STEP 11: Evaluate Hybrid Model on Test Set ───────────────────────────────
leaf_ids_test = dt.apply(X_test_scaled)

# FIX: Use float array — original code used np.zeros_like(y_test) which
# inherits int dtype from pandas Series, silently truncating NN float outputs.
y_hybrid_pred = np.zeros(len(y_test), dtype=np.float64)

for i in tqdm(range(len(X_test_scaled)), desc="Hybrid inference"):
    leaf = leaf_ids_test[i]

    if leaf in leaf_models:
        nn_model       = leaf_models[leaf]
        y_hybrid_pred[i] = nn_model.predict(
            X_test_scaled[i].reshape(1, -1), verbose=0
        )[0, 0]
    else:
        # FIX: was y_dt_test[i] — that variable was never defined!
        y_hybrid_pred[i] = y_pred_test[i]

def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

r2_hybrid     = r2_score(y_test, y_hybrid_pred)
adj_r2_hybrid = adjusted_r2(r2_hybrid, len(y_test), X_test_scaled.shape[1])

print("\n========== FINAL TEST RESULTS ==========")
print(f"DT  Test R²                : {r2_test:.4f}")
print(f"DT  Test Adjusted R²       : {adj_r2_test:.4f}")
print(f"Hybrid Test R²             : {r2_hybrid:.4f}")
print(f"Hybrid Test Adjusted R²    : {adj_r2_hybrid:.4f}")


# ── STEP 12: Save ALL Artifacts ───────────────────────────────────────────────
# Create output directories
os.makedirs("models/leaf_models", exist_ok=True)

# 12a — Decision Tree
joblib.dump(dt, "models/dt_model.pkl")
print("\n✅ Saved: models/dt_model.pkl")

# 12b — StandardScaler
joblib.dump(scaler, "models/scaler.pkl")
print("✅ Saved: models/scaler.pkl")

# 12c — LabelEncoders (all three, keyed by column name)
joblib.dump(encoders, "models/encoders.pkl")
print("✅ Saved: models/encoders.pkl")

# 12d — Leaf NN models (one file per leaf)
for leaf_id, model in leaf_models.items():
    save_path = f"models/leaf_models/{leaf_id}.keras"
    model.save(save_path)
print(f"✅ Saved: models/leaf_models/ ({len(leaf_models)} files)")

# 12e — Leaf metrics JSON
with open("models/leaf_metrics.json", "w") as f:
    json.dump(
        {str(k): v for k, v in leaf_metrics.items()},
        f, indent=2
    )
print("✅ Saved: models/leaf_metrics.json")

# 12f — Config / metadata JSON
config = {
    "model_version":    "1.0",
    "target_column":    TARGET,
    "feature_columns":  FEATURE_COLUMNS,
    "feature_count":    len(FEATURE_COLUMNS),
    "categorical_columns": ["Seasons", "Holiday", "Functioning Day"],
    "min_samples_leaf": MIN_SAMPLES,
    "dt_max_depth":     5,
    "dt_random_state":  42,
    "test_split":       0.2,
    "train_r2":         round(float(r2_train), 4),
    "test_r2":          round(float(r2_test), 4),
    "hybrid_test_r2":   round(float(r2_hybrid), 4),
    "hybrid_adj_r2":    round(float(adj_r2_hybrid), 4),
    "n_leaf_models":    len(leaf_models),
    "seasons_classes":  list(le_seasons.classes_),
    "holiday_classes":  list(le_holiday.classes_),
    "functioning_day_classes": list(le_functioning.classes_),
    "tf_version":       tf.__version__,
}

with open("models/config.json", "w") as f:
    json.dump(config, f, indent=2)
print("✅ Saved: models/config.json")

# ── STEP 13: Final Artifact Summary ──────────────────────────────────────────
print("\n" + "="*55)
print("  ALL ARTIFACTS SAVED SUCCESSFULLY")
print("="*55)
print(f"  models/dt_model.pkl              ← Decision Tree")
print(f"  models/scaler.pkl                ← StandardScaler")
print(f"  models/encoders.pkl              ← 3x LabelEncoders")
print(f"  models/leaf_metrics.json         ← Per-leaf metrics")
print(f"  models/config.json               ← Metadata & classes")
print(f"  models/leaf_models/ ({len(leaf_models):2d} files)   ← Keras leaf NNs")
print("="*55)
print("\n👉 Download the entire 'models/' folder from Colab")
print("   (Files → right-click 'models' → Download)")
