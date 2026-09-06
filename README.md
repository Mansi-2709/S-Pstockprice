# 📈 S&P 500 Stock Price Prediction — LSTM vs XGBoost

## 📌 Project Overview

This project focuses on **S&P 500 stock price prediction using Machine Learning and Deep Learning techniques**.

The primary objective was to compare the performance of two different approaches for time series forecasting:

* **XGBoost Regressor** — Gradient Boosting based Machine Learning model
* **LSTM (Long Short-Term Memory)** — Recurrent Neural Network designed for sequential/time-series data

The models were trained to predict **stock opening prices** and their predictions were evaluated and visually compared against the actual stock prices.

> **Key Finding:** In this experiment, the **LSTM model performed better than XGBoost** for the selected stock/time-series prediction task.

---

## 🎯 Objectives

The main objectives of this project were to:

* Analyze historical S&P 500 stock data
* Perform time-series preprocessing and feature engineering
* Build an **XGBoost regression model**
* Build an **LSTM neural network**
* Generate stock price predictions using both models
* Evaluate model performance using **RMSE**
* Visually compare actual prices with predicted prices
* Determine which model performs better for the given stock prediction problem

---

## 📊 Dataset

The project uses historical S&P 500 stock market data containing information such as:

* Date
* Stock Name
* Open Price
* High Price
* Low Price
* Close Price
* Volume

The dataset contains historical records for multiple S&P 500 companies.

For individual stock prediction, the dataset is filtered based on the selected stock ticker/name before training the models.

---

## 🔄 Project Workflow

```text
Historical S&P 500 Data
          ↓
     Data Cleaning
          ↓
    Date Processing
          ↓
   Select Individual Stock
          ↓
 ┌─────────────────────────┐
 │                         │
 ▼                         ▼
XGBoost Model           LSTM Model
 │                         │
 │ Time-based             │ Scaling
 │ Feature Engineering    │ Sequence Creation
 │ Lag Features           │ 100-day Time Steps
 │                         │
 ▼                         ▼
Predicted Prices       Predicted Prices
 │                         │
 └────────────┬────────────┘
              ↓
       Model Comparison
              ↓
     Actual vs Predicted
              ↓
       Performance Analysis
```

---

# 🤖 Models Used

## 1. XGBoost

The first approach uses **XGBoost Regressor** for stock price forecasting.

Instead of directly feeding the date into the model, several time-based features were extracted.

### Features

The model uses features such as:

* `dayofweek`
* `quarter`
* `month`
* `year`
* `dayofyear`
* `dayofmonth`
* `weekofyear`

In addition, historical **lag features** were created to provide the model with information from previous periods.

### Lag Features

The project incorporates multiple historical observations through lag variables such as:

```text
lag1
lag2
lag3
lag4
lag5
lag6
```

These features allow XGBoost to use historical price information when predicting future prices.

### Time Series Validation

Since stock prices are time-dependent, traditional random train-test splitting was avoided.

**TimeSeriesSplit** was used to maintain the chronological order of the observations during validation.

The XGBoost model was trained using a gradient boosting approach with parameters such as:

* `n_estimators`
* `learning_rate`
* `max_depth`
* `early_stopping_rounds`

---

# 🧠 2. LSTM

The second approach uses an **LSTM neural network**.

LSTM networks are particularly useful for sequential data because they can learn patterns and dependencies across previous observations.

### Preprocessing

The stock opening price was selected as the target variable.

The data was normalized using:

**MinMaxScaler**

This scales the price values to a range between 0 and 1, which is useful when training neural networks.

### Sequence Creation

A **100-day time window** was used.

In other words, the model uses the previous 100 observations to predict the next stock opening price.

```text
Previous 100 observations
          ↓
        LSTM
          ↓
Next predicted price
```

### LSTM Architecture

The model consists of:

```text
Input Sequence
      ↓
LSTM (50 units)
      ↓
LSTM (50 units)
      ↓
LSTM (50 units)
      ↓
Dense Output Layer
      ↓
Predicted Stock Price
```

The model was trained using:

* Optimizer: **Adam**
* Loss Function: **Mean Squared Error**
* Sequence length: **100 time steps**

After prediction, the values were transformed back to their original price scale using the inverse transformation of `MinMaxScaler`.

---

# 📏 Model Evaluation

The models were evaluated using **Root Mean Squared Error (RMSE)**.

RMSE was selected because it measures the magnitude of prediction errors while giving greater weight to larger errors.

The general comparison was:

| Model    | Performance                |
| -------- | -------------------------- |
| XGBoost  | Higher prediction error    |
| **LSTM** | **Lower prediction error** |

### 🏆 Best Performing Model

**LSTM performed better than XGBoost in this experiment.**

The comparison was also visualized using a graph showing:

* Actual stock prices
* XGBoost predictions
* LSTM predictions

This makes it easier to visually understand how closely each model follows the actual price movement.

---

# 📈 Model Comparison Visualization

The project includes visualization of the actual stock price against the predictions generated by both models.

The comparison helps identify:

* How closely predictions follow actual prices
* Where models deviate from actual prices
* How well each model captures price trends
* Which model performs better over the test period

The visual comparison showed that **LSTM was able to follow the underlying time-series pattern more effectively than XGBoost** for this experiment.

---

# 🛠️ Technologies Used

### Programming Language

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* XGBoost

### Deep Learning

* TensorFlow
* Keras

### Data Visualization

* Matplotlib
* Plotly

### Environment

* Google Colab
* Jupyter Notebook

---

# 📚 Key Concepts Demonstrated

This project demonstrates practical understanding of:

* Time Series Forecasting
* Stock Market Data Analysis
* Feature Engineering
* Lag Features
* Time-based Features
* Train/Test Time Series Splitting
* Data Normalization
* Sequence Generation
* LSTM Networks
* XGBoost Regression
* Model Evaluation
* RMSE
* Actual vs Predicted Visualization
* Machine Learning vs Deep Learning Model Comparison

---

# 🔍 XGBoost vs LSTM

| Aspect                      | XGBoost                | LSTM                                             |
| --------------------------- | ---------------------- | ------------------------------------------------ |
| Type                        | Machine Learning       | Deep Learning                                    |
| Architecture                | Gradient Boosted Trees | Recurrent Neural Network                         |
| Sequential Learning         | Indirect               | Direct                                           |
| Feature Engineering         | Required               | Less dependent on manually created time features |
| Lag Features                | Used                   | Learned through sequences                        |
| Scaling                     | Generally not required | Important                                        |
| Time-series sequences       | Explicitly engineered  | Naturally represented                            |
| Performance in this project | Lower                  | **Better**                                       |

---

# 💡 Key Insights

### 1. Time-series data requires chronological validation

Randomly splitting stock data can introduce **data leakage** because future observations may enter the training set.

Using time-based splitting provides a more realistic evaluation.

### 2. Historical information is important

Both approaches use historical information, but they do so differently.

**XGBoost** receives historical information through engineered lag features, while **LSTM** receives sequences of previous observations.

### 3. LSTM performed better in this experiment

The LSTM model achieved better predictive performance than XGBoost based on the evaluation performed in this project.

However, this does **not** mean LSTM will always outperform XGBoost. Performance depends heavily on:

* Stock selected
* Dataset
* Forecast horizon
* Feature engineering
* Hyperparameters
* Train/test period
* Market conditions

---

# ⚠️ Important Disclaimer

This project is intended for **educational and experimental purposes only**.

Stock markets are highly unpredictable and affected by numerous factors including:

* Economic conditions
* Company fundamentals
* News
* Investor sentiment
* Interest rates
* Global events
* Market volatility

Therefore, model predictions should **not be considered financial advice or guaranteed future prices**.

---
# 📁 Project Structure

```text
S-Pstockprice/
│
├── S&P_500_Stock_Price_Prediction.ipynb
│
├── all_stocks_5yr.zip
│
└── README.md
```

> The main analysis, preprocessing, model training, prediction, evaluation, and visualization are contained in the Google Colab notebook.

---

# ▶️ How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/Mansi-2709/S-Pstockprice.git
```

### 2. Open the notebook

Open:

```text
S&P_500_Stock_Price_Prediction.ipynb
```

in **Google Colab** or Jupyter Notebook.

### 3. Install the required libraries

The project requires:

```text
pandas
numpy
scikit-learn
tensorflow
keras
xgboost
matplotlib
plotly
```

### 4. Load the dataset

The historical S&P 500 dataset is provided in the repository.

### 5. Run the notebook

Execute the cells sequentially to:

```text
Load Data
   ↓
Preprocess Data
   ↓
Select Stock
   ↓
Train XGBoost
   ↓
Train LSTM
   ↓
Generate Predictions
   ↓
Calculate RMSE
   ↓
Compare Models
   ↓
Visualize Results
```

---

# 👩‍💻 Project Author

**Mansi Sharma**

Data Analyst / Data Science Enthusiast

### Skills demonstrated through this project

`Python` · `Pandas` · `NumPy` · `SQL` · `Machine Learning` · `Deep Learning` · `XGBoost` · `LSTM` · `Time Series Forecasting` · `Data Visualization`

---

## ⭐ Conclusion

This project explores the application of both **traditional machine learning and deep learning techniques to financial time-series forecasting**.

By implementing and comparing **XGBoost and LSTM**, the project demonstrates how different modeling approaches can handle sequential stock market data.

**The LSTM model achieved better performance than XGBoost in the conducted experiment**, making it the better-performing approach among the two for this particular dataset and prediction setup.
