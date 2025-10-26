import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def make_lstm(
    input_steps: int,
    num_features: int,
    num_classes: int = 2,           # set >2 for multiclass
    lstm_units: int = 64,           # 64 / 48 / 32 per dataset recipe
    dropout_rate: float = 0.3,      # 0.3 / 0.25 / 0.2
    optimizer_name: str = "adam",   # "adam" / "rmsprop"
    lr: float = 3e-4,               # 3e-4 (adam) or 1e-3 (rmsprop)
):
    model = Sequential([
        LSTM(lstm_units, activation='tanh', recurrent_activation='sigmoid',
             input_shape=(input_steps, num_features), recurrent_dropout=0.2),
        Dropout(dropout_rate),
        Dense(num_classes if num_classes>2 else 1,
              activation=('softmax' if num_classes>2 else 'sigmoid'))
    ])
    if optimizer_name == "adam":
        opt = Adam(learning_rate=lr)
    else:
        opt = RMSprop(learning_rate=lr)
    loss = 'categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy'
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy', 'AUC'])
    return model

# Callbacks
early = EarlyStopping(patience=8, restore_best_weights=True, verbose=1)
plateau = ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6, verbose=1)
