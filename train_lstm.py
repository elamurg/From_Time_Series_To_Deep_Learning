# train_lstm.py — W&B sweep-ready trainer for 6 LSTM configs
# (2 case studies × 3 approaches) with optional per-dataset feature overrides.

import os, re, random, sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, callbacks, models
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import wandb

# --- W&B Keras callback (works across wandb versions) ---
try:
    from wandb.integration.keras import WandbCallback
except ImportError:
    from wandb.keras import WandbCallback

# -------------------- Utility: normalize approach names --------------------
def _normalize_approach(val: str) -> str:
    v = str(val).lower().strip()
    mapping = {
        # aliases you may see from older sweeps / UI
        "trend_counts_over_time": "trendcount_only",
        "trend_count_over_time":  "trendcount_only",
        "trend_only":             "trendcount_only",
        "trendcount":             "trendcount_only",

        "merged_data_no_weather": "trendcount_plus_gtrends",
        "merged_data_no weather": "trendcount_plus_gtrends",
        "trend+gtrends":          "trendcount_plus_gtrends",
        "trend_gtrends":          "trendcount_plus_gtrends",

        "merged_data":            "trendcount_gtrends_weather",
    }
    return mapping.get(v, v)

# -------------------- Reproducibility --------------------
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# -------------------- File locations --------------------
# Use "." if your CSVs live next to this script; use "data" if in a /data folder.
DATA_DIR = "."

# -------------------- Feature overrides (optional) --------------------
# If you want different columns per (approach, case_study), put exact names or regex patterns here.
# Leave FEATURE_MAP = {} to use automatic detection based on column keywords.
FEATURE_MAP = {
    # Examples (EDIT or remove these examples):
    # ("trendcount_only", "zara_dress"): [
    #     "zara_dress_trend_count", r"^zara.*trend.*count$"
    # ],
    # ("trendcount_plus_gtrends", "zara_dress"): [
    #     "zara_dress_trend_count", "zara_dress_search_interest"
    # ],
    # ("trendcount_gtrends_weather", "zara_dress"): [
    #     "zara_dress_trend_count", "zara_dress_search_interest",
    #     "avg_temp_c", "rain_mm", "humidity_pct"
    # ],
}

def _resolve_features(df: pd.DataFrame, desired):
    """Expand exact names / regex patterns into existing df columns, preserving order."""
    resolved, seen = [], set()
    cols = df.columns.tolist()
    for item in desired:
        # treat as regex if it looks like one; otherwise try exact then regex
        is_regexy = any(sym in str(item) for sym in "^$[].*+?|()")
        candidates = []
        if not is_regexy and item in cols:
            candidates = [item]
        else:
            try:
                pat = re.compile(str(item))
                candidates = [c for c in cols if pat.search(c)]
            except re.error:
                candidates = [c for c in cols if c == item]
        for c in candidates:
            if c not in seen:
                resolved.append(c); seen.add(c)
    return resolved

# -------------------- CSV reading / normalization --------------------
def _read_csv_safely(pref_names):
    """Try multiple filename variants (handles path or spacing differences)."""
    tried = []
    for name in pref_names:
        path = os.path.join(DATA_DIR, name) if not os.path.isabs(name) else name
        tried.append(path)
        if os.path.exists(path):
            return pd.read_csv(path), path
    raise FileNotFoundError(f"None of these files were found: {tried}")

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"\s+", "_", c.strip().lower()) for c in df.columns]
    # normalize a month/date col if present
    for cand in ["month", "date", "ds", "timestamp"]:
        if cand in df.columns:
            df[cand] = pd.to_datetime(df[cand], errors="coerce")
            break
    return df

# -------------------- Automatic feature detection (fallback) --------------------
def _auto_feature_sets_from_columns(df: pd.DataFrame):
    cols = list(df.columns)
    non_feat = {"month", "date", "ds", "timestamp", "target", "y", "label"}
    cands = [c for c in cols if c not in non_feat]

    is_trendcount = lambda c: ("trend_count" in c) or ("trend" in c and "count" in c)
    is_gtrend     = lambda c: any(k in c for k in ["google_trend", "search_interest", "search_index", "gtrends"])
    is_weather    = lambda c: any(k in c for k in ["weather", "temp", "temperature", "rain", "precip", "humidity", "wind", "snow"])

    trendcount_cols = [c for c in cands if is_trendcount(c)]
    gtrend_cols     = [c for c in cands if is_gtrend(c)]
    weather_cols    = [c for c in cands if is_weather(c)]
    return trendcount_cols, gtrend_cols, weather_cols

def get_feature_columns(cfg, df: pd.DataFrame):
    """Return columns to use as features for this dataset."""
    key = (cfg.approach, cfg.case_study)
    # 1) explicit override map
    if key in FEATURE_MAP and FEATURE_MAP[key]:
        feat_cols = _resolve_features(df, FEATURE_MAP[key])
        if not feat_cols:
            raise ValueError(
                f"No features matched FEATURE_MAP for approach={cfg.approach}, case_study={cfg.case_study}.\n"
                f"Desired: {FEATURE_MAP[key]}\nAvailable: {df.columns.tolist()}"
            )
        return feat_cols

    # 2) auto-detection based on the approach (trendcount / gtrends / weather)
    trendcount_cols, gtrend_cols, weather_cols = _auto_feature_sets_from_columns(df)
    if cfg.approach == "trendcount_only":
        feat_cols = trendcount_cols
    elif cfg.approach == "trendcount_plus_gtrends":
        feat_cols = trendcount_cols + gtrend_cols
    elif cfg.approach == "trendcount_gtrends_weather":
        feat_cols = trendcount_cols + gtrend_cols + weather_cols
    else:
        raise ValueError(f"Unknown approach: {cfg.approach}")

    # de-dup while preserving order
    seen = set()
    feat_cols = [c for c in feat_cols if not (c in seen or seen.add(c))]
    if not feat_cols:
        raise ValueError(
            f"No features detected for approach '{cfg.approach}'. "
            f"Available columns: {df.columns.tolist()}"
        )
    return feat_cols

# -------------------- Dataset loader --------------------
def load_case_study(cfg):
    # normalize approach again (belt & braces)
    cfg.approach = _normalize_approach(cfg.approach)

    # File by approach (support both space and underscore variants)
    if cfg.approach == "trendcount_only":
        pref = ["trend_count_over_time.csv", "data/trend_count_over_time.csv"]
    elif cfg.approach == "trendcount_plus_gtrends":
        pref = ["merged_data_no_weather.csv", "merged_data_no weather.csv",
                "data/merged_data_no_weather.csv", "data/merged_data_no weather.csv"]
    elif cfg.approach == "trendcount_gtrends_weather":
        pref = ["merged_data.csv", "data/merged_data.csv"]
    else:
        raise ValueError(f"Unknown approach: {cfg.approach}")

    raw_df, path = _read_csv_safely(pref)
    df = _normalize_columns(raw_df)

    # pick target by case study (EDIT these if your target names differ)
    if cfg.case_study == "zara_dress":
        targets_try = ["zara_dress", "target_zara_dress", "target", "y"]
    elif cfg.case_study == "chanel_bag":
        targets_try = ["chanel_bag", "target_chanel_bag", "target", "y"]
    else:
        raise ValueError(f"Unknown case_study: {cfg.case_study}")

    target_col = next((t for t in targets_try if t in df.columns), None)
    if target_col is None:
        raise ValueError(
            f"Target column not found for {cfg.case_study}. "
            f"Tried: {targets_try}. Columns: {df.columns.tolist()}"
        )

    # sort to monthly start frequency like your original
    date_col = None
    for cand in ["month", "date", "ds", "timestamp"]:
        if cand in df.columns:
            date_col = cand
            break
    if date_col:
        df = df.sort_values(date_col).set_index(date_col).asfreq("MS").reset_index()

    feat_cols = get_feature_columns(cfg, df)
    df = df.dropna(subset=feat_cols + [target_col]).reset_index(drop=True)

    return df, feat_cols, target_col, date_col, path

# -------------------- Sequences & splits --------------------
def create_sequences(X, y, seq_length=6):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length])
    return np.array(X_seq), np.array(y_seq)

def chronological_split(X, y, val_ratio=0.2):
    n = len(X)
    n_val = int(n * val_ratio)
    n_train = max(0, n - n_val)
    return (X[:n_train], y[:n_train], X[n_train:], y[n_train:])

# -------------------- Model --------------------
def build_model(cfg, n_features):
    model = models.Sequential()
    model.add(layers.LSTM(
        int(cfg.lstm_units),
        activation=getattr(cfg, "lstm_activation", "tanh"),
        return_sequences=bool(cfg.stacked),
        input_shape=(int(cfg.lookback), n_features)
    ))
    model.add(layers.Dropout(float(cfg.dropout)))
    if bool(cfg.stacked):
        model.add(layers.LSTM(
            max(32, int(cfg.lstm_units) // 2),
            activation=getattr(cfg, "lstm_activation", "tanh")
        ))
        model.add(layers.Dropout(float(cfg.dropout)))
    model.add(layers.Dense(1, activation=getattr(cfg, "dense_activation", "relu")))

    # optimizer
    if cfg.optimizer == "adam":
        opt = tf.keras.optimizers.Adam(learning_rate=float(cfg.learning_rate))
    elif cfg.optimizer == "nadam":
        opt = tf.keras.optimizers.Nadam(learning_rate=float(cfg.learning_rate))
    elif cfg.optimizer == "rmsprop":
        opt = tf.keras.optimizers.RMSprop(learning_rate=float(cfg.learning_rate))
    else:
        opt = tf.keras.optimizers.Adam(learning_rate=float(cfg.learning_rate))

    loss = "mse" if cfg.loss == "mse" else tf.keras.losses.Huber()
    model.compile(optimizer=opt, loss=loss, metrics=["mae"])
    return model

# -------------------- Main --------------------
def main():
    # 1) Init first (don’t read wandb.config inside init args)
    run = wandb.init(
        project="LSTM_publication",          # W&B project name
        tags=["lstm", "timeseries"],
        config={
            # categorical families (the sweep will override these)
            "case_study": "zara_dress",      # ["zara_dress","chanel_bag"]
            "approach": "trendcount_only",   # ["trendcount_only","trendcount_plus_gtrends","trendcount_gtrends_weather"]

            # model/search defaults (sweep overrides many of these)
            "lookback": 6,
            "lstm_units": 64,
            "dropout": 0.1,
            "stacked": False,
            "optimizer": "adam",
            "learning_rate": 1e-3,
            "loss": "mse",
            "epochs": 80,
            "batch_size": 16,
            "val_ratio": 0.2,

            # activation search
            "lstm_activation": "tanh",
            "dense_activation": "relu",

            # optional: manual feature override list (exact/regex strings)
            "feature_overrides": [],
        },
    )

    cfg = wandb.config

    # 2) Normalize approach names if sweep passes aliases
    norm = _normalize_approach(cfg.approach)
    if norm != cfg.approach:
        wandb.config.update({"approach": norm}, allow_val_change=True)
        cfg = wandb.config

    # 3) Grouping by family (case_study_approach)
    group = f"{cfg.get('case_study','?')}_{cfg.get('approach','?')}"
    try:
        run.group = group
    except Exception:
        pass
    wandb.log({"group_name": group})

    # 4) Load data
    df, feat_cols, target_col, date_col, used_path = load_case_study(cfg)
    wandb.log({
        "data_file_used": used_path,
        "target_col": target_col,
        "features_used": ",".join(feat_cols),
    })

    # 5) Scale (fit on train only)
    X = df[feat_cols].values
    y = df[[target_col]].values  # column vector

    X_train_raw, y_train_raw, X_val_raw, y_val_raw = chronological_split(X, y, val_ratio=float(cfg.val_ratio))

    if len(X_train_raw) < int(cfg.lookback) + 1 or len(X_val_raw) < int(cfg.lookback) + 1:
        print("Not enough data after split for the chosen lookback; skipping run.", file=sys.stderr)
        wandb.log({"status": "insufficient_data"})
        run.finish()
        return

    scaler_X = MinMaxScaler().fit(X_train_raw)
    scaler_y = MinMaxScaler().fit(y_train_raw)

    X_train_scaled = scaler_X.transform(X_train_raw)
    y_train_scaled = scaler_y.transform(y_train_raw)
    X_val_scaled   = scaler_X.transform(X_val_raw)
    y_val_scaled   = scaler_y.transform(y_val_raw)

    # 6) Build sequences
    lookback = int(cfg.lookback)
    Xtr_seq, ytr_seq = create_sequences(X_train_scaled, y_train_scaled, lookback)
    Xv_seq,  yv_seq  = create_sequences(X_val_scaled,   y_val_scaled,   lookback)

    if len(Xtr_seq) == 0 or len(Xv_seq) == 0:
        print("No sequences produced (check lookback vs data length).", file=sys.stderr)
        wandb.log({"status": "no_sequences"})
        run.finish()
        return

    # 7) Model
    model = build_model(cfg, n_features=Xtr_seq.shape[-1])

    cbs = [
        callbacks.EarlyStopping(monitor="val_mae", patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_mae", factor=0.5, patience=5, min_lr=1e-6),
        callbacks.ModelCheckpoint("model-best.keras", monitor="val_mae", save_best_only=True),
        WandbCallback(save_model=False),
    ]

    wandb.log({
        "lstm_activation": getattr(cfg, "lstm_activation", "tanh"),
        "dense_activation": getattr(cfg, "dense_activation", "relu"),
    })

    history = model.fit(
        Xtr_seq, ytr_seq,
        validation_data=(Xv_seq, yv_seq),
        epochs=int(cfg.epochs),
        batch_size=int(cfg.batch_size),
        callbacks=cbs,
        verbose=1,
    )

    # 8) Evaluate (inverse-scale)
    yhat_v_scaled = model.predict(Xv_seq).ravel()
    yhat_v = scaler_y.inverse_transform(yhat_v_scaled.reshape(-1, 1)).ravel()
    yv_true = scaler_y.inverse_transform(yv_seq.reshape(-1, 1)).ravel()

    rmse = mean_squared_error(yv_true, yhat_v, squared=False)
    mae  = mean_absolute_error(yv_true, yhat_v)
    mape = float(np.mean(np.abs((yv_true - yhat_v) / np.clip(np.abs(yv_true), 1e-6, None))) * 100.0)

    wandb.log({"val_rmse": rmse, "val_mae": mae, "val_mape": mape})
    wandb.save("model-best.keras")
    run.finish()

if __name__ == "__main__":
    main()
