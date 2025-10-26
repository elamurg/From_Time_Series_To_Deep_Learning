import pandas as pd

datasets = {
    "merged_data": pd.read_csv("merged_data.csv"),
    "merged_no_weather": pd.read_csv("merged_data_no_weather.csv"),
    "trend_counts": pd.read_csv("trend_counts_over_time.csv")
} #dictionary loading csv into dataframe

configs = {
    "merged_data": {"units": 64, "dropout": 0.3, "optimizer": "adam", "lr": 3e-4},
    "merged_no_weather": {"units": 48, "dropout": 0.25, "optimizer": "adam", "lr": 3e-4},
    "trend_counts": {"units": 32, "dropout": 0.2, "optimizer": "rmsprop", "lr": 1e-3}
} #setting configurations for each dataset
#nmbr of neurons in the LSTM layer, percentage of neurons randomly disable to avoid iverfitting, adjusting model weights w adam and rmsprop, and learning rate

items = ["zara dress", "chanel bag"]

for name, df in datasets.items(): #iterating through each dataset and matching it w its configuration
    cfg = configs[name]

    for item in items:
        print(f"Training model for {item} using {name} dataset")

        item_df = df[df['keyword_column'].str.contains(item, case=False, na=False)] #filtering the dataframe to only include rows where the keyword_column contains the item name, case insensitive, na values are ignored

        X, y = make_sequences(item_df, seq_length=52)  # a helper function you’d define

        model = make_lstm(
            input_steps=X.shape[1],
            num_features=X.shape[2],
            lstm_units=cfg["units"],
            dropout_rate=cfg["dropout"],
            optimizer_name=cfg["optimizer"],
            lr=cfg["lr"]
        )

        model.fit(X, y, validation_split=0.2, epochs=100, batch_size=32, callbacks=[early, plateau])
