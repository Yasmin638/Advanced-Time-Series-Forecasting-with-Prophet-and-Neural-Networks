# **Advanced Time-Series Forecasting with Prophet and Neural Networks**

This project aims to build an advanced machine-learning–based forecasting system using **Facebook Prophet** and **LSTM Neural Networks**. The goal is to model and predict synthetic time-series data that include trend, seasonality, noise, and external regressors.
The project demonstrates:

* Synthetic dataset generation
* Feature engineering
* Deep learning forecasting
* Prophet model tuning
* Rolling-origin cross-validation
* Detailed model comparison

This work highlights how traditional statistical forecasting and deep-learning models differ in performance under various conditions and helps identify which model is more suitable for different time-series patterns.



## **📊 Data**

The dataset used for training and evaluation was **synthetically generated** to simulate realistic signals found in financial, energy, climate, and agricultural time-series forecasting.

The dataset includes:

* **Daily timestamps**
* **Trend component**
* **Weekly seasonality**
* **Noise**
* **External regressor** (e.g., synthetic temperature-like feature)

The synthetic dataset enables full control over features, making it easier to evaluate model behavior, strengths, and weaknesses.

### **Data Features**

* `date`: Timestamp
* `y`: Target variable
* `regressor`: External variable added to Prophet
* `lag features`: Used for LSTM windowing

### **Visualizations Included**

* Trend visualization
* Seasonality patterns
* Prophet component plots
* LSTM prediction vs actual
* Forecast comparison chart
* RMSE / MAE bar charts



## **🔍 Methodology**

### **1. Synthetic Dataset Generation**

* Created daily time-series containing:

  * Linear trend
  * Weekly seasonality
  * Random noise
  * Additional regressor
* Exported data for cross-validation and testing.

### **2. Preprocessing**

* Converted Prophet-compatible dataframe
* Created windowed sequences for LSTM
* Normalized features
* Split into training and test sets

### **3. Prophet Model**

* Added regressor
* Tuning:

  * Changepoint range
  * Seasonality mode
  * Growth type
* 7-day ahead forecasting

### **4. LSTM Neural Network**

* Used sequence windowing
* Tuned:

  * Hidden units
  * Learning rate
  * Batch size
* Evaluated using RMSE, MAE, and MAPE

### **5. Rolling-Origin Cross-Validation**

Performed repeated walk-forward splits to simulate real-world forecasting conditions.


## **📈 Results**

* **Prophet** performed better on strong seasonal + trend patterns.
* **LSTM** showed stronger accuracy on noisy and non-linear patterns.
* Rolling-origin cross-validation confirmed stable error metrics.

### **Outputs Generated**

* Forecast CSV
* Forecast summary CSV
* Text-based model report
* Side-by-side forecast plots



## **📁 Project Structure**

```
.
├── main.py
├── README.md
├── LICENSE
├── .gitignore
└── Advanced_Time_Series_Forecasting_with_Prophet_and_LSTM.ipynb
```



## **🚀 How to Run**

```bash
pip install -r requirements.txt
python main.py
```



## **📄 Conclusion**

This project demonstrates a complete and modern forecasting workflow using both statistical and deep-learning methods. It shows how incorporating external regressors, tuning, and rolling validation can lead to more reliable predictions—useful for finance, weather prediction, energy load forecasting, agriculture, and more.


