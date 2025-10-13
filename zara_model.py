import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential  # model type
from tensorflow.keras.layers import LSTM, Dense  # layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

seed = 42 #reproductability across runs and systems
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

df = pd.read_csv("merged_data_no_weather.csv")
df['Month'] = pd.to_datetime(df['Month']) #converts to proper datetime format
df.set_index('Month', inplace = True)
df =df.asfreq('MS') #consistent monthly frequency

features = ['Zara Dress Search Interest','Chanel Bag Search Interest']
target = 'zara dress'

cutoff_date = pd.Timestamp('2013-10-01')
train_df = df[df.index < cutoff_date] #learns the patterns before the cutoff date
test_df = df[df.index >= cutoff_date] #test preformance on data its never seen

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

#Scaling the feature and target variable, neural networks perform best when values are small
X_train_scaled = scaler_X.fit_transform(train_df[features]) #fit_transform is used for training data, learns scaling parameters from training data and applies the tranformation
y_train_scaled = scaler_y.fit_transform(train_df[[target]])
X_test_scaled = scaler_X.transform(test_df[features]) #transform is used to test data, this applies the same scaling from the training data and it prevents information to leak into the training process
y_test_scaled = scaler_y.transform(test_df[[target]])

#creating sequences for LSTM input
def create_sequences(X, y, seq_length=6): #sliding window of 6 months across all data
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i + seq_length]) #creates the input sequence (historical data) 6 months of data and adds it to our list of inputs
        y_seq.append(y[i + seq_length]) #target value (the correct answer for the input sequence)
    return np.array(X_seq), np.array(y_seq)
seq_length = 6
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, seq_length)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, seq_length)

#LSTM's input shape
model = Sequential()
model.add(LSTM(64, activation='tanh', input_shape=(seq_length, len(features))))
model.add(Dense(1)) #final output layer, fully-connected layer with a single neuron as we are predicting a single value
model.compile(optimizer='adam', loss ='mse') #adam optimizer for updating the model'S internal weights
model.fit(X_train_seq, y_train_seq, epochs = 100, batch_size = 8, verbose = 1) #the model will go though the entire dataset 100 times, and process data in small groups of 8 sequences

y_pred_scaled = model.predict(X_test_seq) #makes predictions on the test sequence
y_pred = scaler_y.inverse_transform(y_pred_scaled) #scaling model predictions back to original units
y_test_actual = scaler_y.inverse_transform(y_test_seq)

forecast_dates = test_df.index[seq_length : seq_length +len(y_pred)]
#plt.figure(figsize=(12,5))
#plt.plot(df.index, df[target], label='Actual Data', color='black')
#plt.plot(forecast_dates, y_pred.flatten(), label='LSTM Forecast (tanh/adam optimizer)', linestyle='--')
#plt.title(f"LSTM Forecast: {target.title()}")
#plt.xlabel("Date")
#plt.ylabel("Trend Count")
#plt.legend()
#plt.grid(True)
#plt.tight_layout()
#plt.show()

mse = mean_squared_error(y_test_actual, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_actual, y_pred)
mape = np.mean(np.abs((y_test_actual - y_pred) / y_test_actual)) * 100



print("Model Evaluation with Alternative Parameters:")
print(f"LSTM RMSE: {rmse:.2f}")
print(f"LSTM MAE: {mae:.2f}")
print(f"LSTM MAPE: {mape:.2f}%")

