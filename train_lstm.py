import os, re, random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, callbacks, models
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import wandb
from wandb.integration.keras import WandbCallback

# -------------------- Reproducibility --------------------
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# -------------------- Data mapping & feature picking --------------------
DATA_DIR = "."

def _read_csv_safely(pref_names):
    """Try multiple filename variants (handles space vs underscore)."""
    for name in pref_names:
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path):
            df = pd.read_csv(path)
            return df, path
    raise FileNotFoundError(f"None of these files were found: {pref_names}")

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"\s+", "_", c.strip().lower()) for c in df.columns]
    # normalize a month/date col if present
    for cand in ["month", "date", "ds", "timestamp"]:
        if cand in df.columns:
            df[cand] = pd.to_datetime(df[cand], errors="coerce")
            break
    return df

def _feature_sets_from_columns(df: pd.DataFrame):
    cols = list(df.columns)
    non_feat = {"month", "date", "ds", "timestamp", "target", "y", "label"}
    cands = [c for c in cols if c not in non_feat]
    # Keywords tuned to your headers
    is_trendcount = lambda c: ("trend_count" in c) or ("count" in c and "trend" in c)
    is_gtrend     = lambda c: any(k in c for k in ["google_trend", "search_interest", "search_index", "gtrends"])
    is_weather    = lambda c: any(k in c for k in ["weather", "temp", "temperature", "rain", "precip", "humidity", "wind", "snow"])

    trendcount_cols = [c for c in cands if is_trendcount(c) or ("trend" in c and "count" in c)]
    gtrend_cols     = [c for c in cands if is_gtrend(c)]
    weather_cols    = [c for c in cands if is_weather(c)]
    return trendcount_cols, gtrend_cols, weather_cols

def get_feature_columns(cfg, df: pd.DataFrame):
    trendcount_cols, gtrend_cols, weather_cols = _feature_sets_from_columns(df)
    if cfg.approach == "trendcount_only":
        feat_cols = trendcount_cols
    elif cfg.approach == "trendcount_plus_gtrends":
        feat_cols = trendcount_cols + gtrend_cols
    elif cfg.approach == "trendcount_gtrends_weather":
        feat_cols = trendcount_cols + gtrend_cols + weather_cols
    else:
        raise ValueError(f"Unknown approach: {cfg.approach}")
    # dedupe preserve order
    seen = set()
    feat_cols = [c for c in feat_cols if not (c in seen or seen.add(c))]
    if not feat_cols:
        raise ValueError(f"No features detected for approach '{cfg.approach}'. Columns: {df.columns.tolist()}")
    return feat_cols

def load_case_study(cfg):
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

    # pick target by case study (adjust names if yours differ)
    if cfg.case_study == "zara_dress":
        targets_try = ["zara_dress", "target_zara_dress", "target", "y"]
    elif cfg.case_study == "chanel_bag":
        targets_try = ["chanel_bag", "target_chanel_bag", "target", "y"]
    else:
        raise ValueError(f"Unknown case_study: {cfg.case_study}")

    target_col = next((t for t in targets_try if t in df.columns), None)
    if target_col is None:
        raise ValueError(f"Target column not found for {cfg.case_study}. Tried {targets_try}. Columns: {df.columns.tolist()}")

    # date/month ordering
    date_col = None
    for cand in ["month", "date", "ds", "timestamp"]:
        if cand in df.columns:
            date_col = cand
            break
    if date_col:
        # match your original: monthly start frequency
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
    n_train = n - n_val
    return (X[:n_train], y[:n_train], X[n_train:], y[n_train:])

# -------------------- Model --------------------
def build_model(cfg, n_features):
    model = models.Sequential()
    model.add(layers.LSTM(cfg.lstm_units, activation="tanh",
                          return_sequences=cfg.stacked,
                          input_shape=(cfg.lookback, n_features)))
    model.add(layers.Dropout(cfg.dropout))
    if cfg.stacked:
        model.add(layers.LSTM(max(32, cfg.lstm_units // 2)))
        model.add(layers.Dropout(cfg.dropout))
    model.add(layers.Dense(1, activation="relu"))

    # optimizer
    if cfg.optimizer == "adam":
        opt = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)
    elif cfg.optimizer == "nadam":
        opt = tf.keras.optimizers.Nadam(learning_rate=cfg.learning_rate)
    elif cfg.optimizer == "rmsprop":
        opt = tf.keras.optimizers.RMSprop(learning_rate=cfg.learning_rate)
    else:
        opt = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)

    loss = "mse" if cfg.loss == "mse" else tf.keras.losses.Huber()
    model.compile(optimizer=opt, loss=loss, metrics=["mae"])
    return model

# -------------------- Main --------------------
def main():
    wandb.init(
        project="fashion-trend-lstm",
        tags=["lstm","timeseries"],
        group=f"{wandb.config.get('case_study','?')}_{wandb.config.get('approach','?')}",
        config={
            "case_study": "zara_dress",  # ["zara_dress","chanel_bag"]
            "approach": "trendcount_only",  # ["trendcount_only","trendcount_plus_gtrends","trendcount_gtrends_weather"]
            "lookback": 6,
            "lstm_units": 64,
            "dropout": 0.1,
            "stacked": False,
            "optimizer": "adam",
            "learning_rate": 1e-3,
            "loss": "mse",
            "epochs": 80,
            "batch_size": 16,
            "val_ratio": 0.2
        }
    )
    cfg = wandb.config

    # ----- load data -----
    df, feat_cols, target_col, date_col, used_path = load_case_study(cfg)
    wandb.log({"data_file_used": used_path})

    # ----- scale with MinMax (match your LSTM_model.py) -----
    X = df[feat_cols].values
    y = df[[target_col]].values  # column vector

    # chronological split first, then fit scalers on train only
    X_train_raw, y_train_raw, X_val_raw, y_val_raw = chronological_split(X, y, val_ratio=cfg.val_ratio)

    scaler_X = MinMaxScaler().fit(X_train_raw)
    scaler_y = MinMaxScaler().fit(y_train_raw)

    X_train_scaled = scaler_X.transform(X_train_raw)
    y_train_scaled = scaler_y.transform(y_train_raw)
    X_val_scaled   = scaler_X.transform(X_val_raw)
    y_val_scaled   = scaler_y.transform(y_val_raw)

    # sequences
    lookback = int(cfg.lookback)
    Xtr_seq, ytr_seq = create_sequences(X_train_scaled, y_train_scaled, lookback)
    Xv_seq,  yv_seq  = create_sequences(X_val_scaled,   y_val_scaled,   lookback)

    # ----- model -----
    model = build_model(cfg, n_features=Xtr_seq.shape[-1])

    cbs = [
        callbacks.EarlyStopping(monitor="val_mae", patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_mae", factor=0.5, patience=5, min_lr=1e-6),
        callbacks.ModelCheckpoint("model-best.keras", monitor="val_mae", save_best_only=True),
        WandbCallback(save_model=False)
    ]

    history = model.fit(
        Xtr_seq, ytr_seq,
        validation_data=(Xv_seq, yv_seq),
        epochs=int(cfg.epochs),
        batch_size=int(cfg.batch_size),
        callbacks=cbs,
        verbose=1
    )

    # ----- evaluate on validation (inverse-scale) -----
    yhat_v_scaled = model.predict(Xv_seq).ravel()
    yhat_v = scaler_y.inverse_transform(yhat_v_scaled.reshape(-1, 1)).ravel()
    yv_true = scaler_y.inverse_transform(yv_seq.reshape(-1, 1)).ravel()

    rmse = mean_squared_error(yv_true, yhat_v, squared=False)
    mae  = mean_absolute_error(yv_true, yhat_v)
    mape = float(np.mean(np.abs((yv_true - yhat_v) / np.clip(np.abs(yv_true), 1e-6, None))) * 100.0)

    wandb.log({"val_rmse": rmse, "val_mae": mae, "val_mape": mape})
    wandb.save("model-best.keras")
    wandb.finish()

if __name__ == "__main__":
    main()
