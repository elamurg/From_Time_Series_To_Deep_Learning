# From Time Series to Deep Learning: Forecasting Fashion Trend Lifecycles with LSTM

This project explores the use of **Long Short-Term Memory (LSTM)** neural networks to predict fashion trend lifecycles using **Google Trends data** and **weather data**. Specifically, it models search interest over time for key fashion items (e.g., *Zara dresses* and *Chanel bags*) to identify trend saturation patterns and evaluate the predictive performance of deep learning on temporal fashion data.

---

## Project Overview

The primary goal of this research is to **predict the future popularity of fashion trends** based on past patterns and cross-variable relationships between different trend indicators. The model captures temporal dependencies through sequential data and evaluates its accuracy using various error metrics.

This study contributes to **AI-driven fashion forecasting**, bridging data science and trend analysis to enhance predictive accuracy in identifying when trends peak and decline.

---

## Methodology

### 1. Data Preparation

* **Dataset:** `merged_data_no_weather.csv` or `merged_data.csv` or `trend_counts_over_time.csv` 
* **Variables:**

Month,Zara Dress Search Interest,Chanel Bag Search Interest,avg_monthly_temp,avg_monthly_rain,zara dress,chanel bag
  * Features: `Zara Dress Search Interest`, `Chanel Bag Search Interest`, `avg_monthly_temp`, `avg_monthly_rain` or `Zara Dress Search Interest`, `Chanel Bag Search Interest` or `zara dress`, `chanel bag`
  * Target: `zara dress` or `chanel bag`
* **Frequency:** Monthly (`MS`)
* **Training/Test Split:** Based on cutoff date `2013-10-01`

### 2. Preprocessing Steps

* Converted timestamps to datetime objects
* Resampled to ensure a consistent monthly frequency
* Scaled features and targets using `MinMaxScaler` to optimize neural network training
* Created time sequences (6 months sliding window) for LSTM input

### 3. Model Architecture

```python
model = Sequential()
model.add(LSTM(64, activation='tanh', input_shape=(6, 2)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
```

* **LSTM layer:** 64 units, `tanh` activation
* **Output layer:** 1 neuron for single-value prediction
* **Optimizer:** Adam
* **Loss:** Mean Squared Error (MSE)
* **Epochs:** 100
* **Batch size:** 8

---

## Evaluation Metrics

After training, the model’s predictions are compared with actual data using:

| Metric | Description                    | Formula            |         |             |
| :----- | :----------------------------- | :----------------- | ------- | ----------- |
| RMSE   | Root Mean Squared Error        | √(Σ(yᵢ - ŷᵢ)² / n) |         |             |
| MAE    | Mean Absolute Error            | Σ                  | yᵢ - ŷᵢ | / n         |
| MAPE   | Mean Absolute Percentage Error | (Σ                 | yᵢ - ŷᵢ | / yᵢ) * 100 |

**Results Example:**

```
Model Evaluation with Alternative Parameters:
LSTM RMSE: 3.82
LSTM MAE: 2.91
LSTM MAPE: 8.43%
```

---

## Visualization

Although commented out in the main script, the visualization code below can be enabled to plot actual vs. predicted trend trajectories:

```python
plt.figure(figsize=(12,5))
plt.plot(df.index, df[target], label='Actual Data', color='black')
plt.plot(forecast_dates, y_pred.flatten(), label='LSTM Forecast', linestyle='--')
plt.title(f"LSTM Forecast: {target.title()}")
plt.xlabel("Date")
plt.ylabel("Trend Count")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

---

## Key Dependencies

* Python 3.10+
* TensorFlow
* NumPy
* Pandas
* Scikit-learn
* Matplotlib

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## Reproducibility

To ensure consistent results across runs:

```python
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)
```

---

## Research Context

This work forms part of a larger dissertation investigating **AI-driven vs. traditional forecasting models** in predicting **fashion trend saturation stages**. The LSTM model presented here represents the deep learning component, offering insight into how sequence models can capture nonlinear temporal relationships in fashion-related search data.

---

## Citation

If you use this repository for academic or professional work, please cite as:

```
Murgelj, E. (2025). From Time Series Statistics to Deep Learning: Forecasting Saturation in Fashion Trend Lifecycles using a Multimodal Approach. 
GitHub Repository. https://github.com/elamurg/From_Time_Series_To_Deep_Learning
```

---

## Author

**Ela Murgelj**
MSc Computer Science| Machine Learning Engineer | AI & Trend Forecasting Enthusiast
[GitHub](https://github.com/elamurg) | [LinkedIn](https://www.linkedin.com/in/elamurgelj)

