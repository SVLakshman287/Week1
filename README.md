# ☀️ Solar Power Prediction

> **Predict solar power generation using Machine Learning & Linear Regression**
>
> A comprehensive data science project that forecasts solar power output based on meteorological and solar radiation data.

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat-square&logo=jupyter)](https://jupyter.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-green?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)]()
[![Dataset](https://img.shields.io/badge/Dataset-4.2K%20Records-blue?style=flat-square)]()

</div>

---

## 📋 Table of Contents

- [🎯 Overview](#-overview)
- [📊 Dataset](#-dataset)
- [🔍 Features](#-features)
- [🤖 Model Architecture](#-model-architecture)
- [📁 Project Structure](#-project-structure)
- [📦 Requirements & Dependencies](#-requirements--dependencies)
- [🚀 Installation](#-installation)
- [💻 Usage Guide](#-usage-guide)
- [📈 Results & Performance](#-results--performance)
- [🔮 Future Improvements](#-future-improvements)
- [🎓 Learning Outcomes](#-learning-outcomes)
- [📚 References](#-references)
- [🤝 Contributing](#-contributing)
- [📝 License](#-license)
- [👤 Author](#-author)

---

## 🎯 Overview

This project demonstrates a **complete machine learning workflow** for predicting solar power generation. It serves as an excellent educational resource for understanding:

- ✅ Data loading, exploration, and analysis
- ✅ Feature engineering and preprocessing
- ✅ Model training and evaluation
- ✅ Performance metrics and visualization
- ✅ Real-world renewable energy applications

### Key Highlights

🔋 **Energy Prediction** - Forecast solar power output with machine learning

🌍 **Real-World Application** - Applicable to renewable energy management

📊 **21 Features** - Meteorological, solar, and wind data combined

⚡ **4,213 Records** - Comprehensive historical dataset

🎯 **Linear Regression** - Simple, interpretable, and effective baseline model

---

## 📊 Dataset

### Dataset Overview

| Property | Details |
|----------|---------|
| 📁 **File Name** | `solarpowergeneration.csv` |
| 📦 **File Size** | ~498 KB |
| 📝 **Total Records** | 4,213 observations |
| 🔢 **Features** | 21 variables |
| 🎯 **Target Variable** | `generated_power_kw` |
| 🌡️ **Temperature Range** | -5.35°C to 34.9°C |
| ⚡ **Power Output Range** | 0.0006 kW to 3,056.79 kW |
| 📈 **Average Power** | 1,134.35 kW |
| 📊 **Std Deviation** | 937.96 kW |

### Sample Data

```
   temperature_2_m_above_gnd  relative_humidity_2_m_above_gnd  mean_sea_level_pressure_MSL  ...  generated_power_kw
0                       2.17                               31                       1035.0  ...           454.10
1                       2.31                               27                       1035.1  ...          1411.99
2                       3.65                               33                       1035.4  ...          2214.85
3                       5.82                               30                       1035.4  ...          2527.61
4                       7.73                               27                       1034.4  ...          2640.20
```

---

## 🔍 Features

The dataset contains **21 features** across 5 categories:

### 🌡️ Meteorological Data (5 Features)

| Feature | Description | Unit | Range |
|---------|-------------|------|-------|
| `temperature_2_m_above_gnd` | Air temperature at 2m height | °C | -5.35 to 34.9 |
| `relative_humidity_2_m_above_gnd` | Relative humidity at 2m height | % | 7 to 100 |
| `mean_sea_level_pressure_MSL` | Mean sea level pressure | hPa | 997.5 to 1,046.8 |
| `total_precipitation_sfc` | Total precipitation | mm | 0 to 3.2 |
| `snowfall_amount_sfc` | Snowfall amount | mm | 0 to 1.68 |

### ☁️ Cloud Cover Data (4 Features)

| Feature | Description | Unit | Range |
|---------|-------------|------|-------|
| `total_cloud_cover_sfc` | Total cloud cover | % | 0 to 100 |
| `high_cloud_cover_high_cld_lay` | High altitude clouds | % | 0 to 100 |
| `medium_cloud_cover_mid_cld_lay` | Medium altitude clouds | % | 0 to 100 |
| `low_cloud_cover_low_cld_lay` | Low altitude clouds | % | 0 to 100 |

### ☀️ Solar Radiation Data (1 Feature)

| Feature | Description | Unit | Range |
|---------|-------------|------|-------|
| `shortwave_radiation_backwards_sfc` | Downward shortwave radiation | W/m² | 0 to 952.3 |

### 💨 Wind Data (7 Features)

| Feature | Description | Unit | Range |
|---------|-------------|------|-------|
| `wind_speed_10_m_above_gnd` | Wind speed at 10m height | m/s | 0 to 66.88 |
| `wind_direction_10_m_above_gnd` | Wind direction at 10m | ° | 0.54 to 360 |
| `wind_speed_80_m_above_gnd` | Wind speed at 80m height | m/s | 0 to 66.88 |
| `wind_direction_80_m_above_gnd` | Wind direction at 80m | ° | 1.12 to 360 |
| `wind_speed_900_mb` | Wind speed at 900 mb pressure | m/s | 0 to 61.11 |
| `wind_direction_900_mb` | Wind direction at 900 mb | ° | 1.12 to 360 |
| `wind_gust_10_m_above_gnd` | Wind gust speed at 10m | m/s | 0.72 to 84.96 |

### 🔆 Solar Geometry Data (3 Features)

| Feature | Description | Unit | Range |
|---------|-------------|------|-------|
| `angle_of_incidence` | Solar angle vs panel surface | ° | 3.76 to 121.64 |
| `zenith` | Solar zenith angle | ° | 17.73 to 128.42 |
| `azimuth` | Solar azimuth angle | ° | 54.38 to 289.05 |

### ⚡ Target Variable (1 Feature)

| Feature | Description | Unit | Range |
|---------|-------------|------|-------|
| `generated_power_kw` | Solar power output | kW | 0.0006 to 3,056.79 |

---

## 🤖 Model Architecture

### Algorithm: Linear Regression

**Linear Regression** models the relationship between input features and target output as:

```
Power Output = β₀ + β₁·X₁ + β₂·X₂ + ... + β₂₁·X₂₁ + ε
```

Where:
- 🎯 **Power Output** = Predicted solar power generation
- 📊 **X₁ to X₂₁** = Input features (21 features)
- 📐 **β₀ to β₂₁** = Learned coefficients
- ⚠️ **ε** = Error term

### Model Workflow

```
Input Data (4,213 records)
         ↓
Data Exploration & Analysis
         ↓
Feature Preprocessing
         ↓
Train-Test Split (80-20)
         ↓
Linear Regression Training
         ↓
Model Evaluation
         ↓
Power Output Predictions
```

### Why Linear Regression?

✅ **Interpretability** - Clear relationship between features and output

✅ **Performance** - Fast training and inference

✅ **Baseline** - Establishes comparison benchmark for advanced models

✅ **Simplicity** - Fewer hyperparameters to tune

✅ **Explainability** - Easy to understand coefficients

---

## 📁 Project Structure

```
Week1/
│
├── 📔 soloarpowerprediction.ipynb              Main Jupyter Notebook
│   ├── Data Loading & Exploration
│   ├── Exploratory Data Analysis (EDA)
│   ├── Data Preprocessing
│   ├── Model Training
│   ├── Model Evaluation
│   └── Predictions & Insights
│
├── 📊 solarpowergeneration.csv                  Dataset
│   └── 4,213 records × 21 features
│
├── 📽️ Solar_Power_Prediction_Presentation.pptx  Results Presentation
│   └── Slides with findings & visualizations
│
├── 📝 README.md                                 Documentation
│
└── 📦 requirements.txt                          Python Dependencies
```

### File Descriptions

| File | Type | Size | Purpose |
|------|------|------|---------|
| `soloarpowerprediction.ipynb` | Jupyter | 2.1 MB | Complete ML pipeline with analysis & training |
| `solarpowergeneration.csv` | CSV Data | 498 KB | Historical solar power generation data |
| `Solar_Power_Prediction_Presentation.pptx` | PowerPoint | 2.7 MB | Presentation with results & visualizations |
| `README.md` | Markdown | - | Project documentation |

---

## 📦 Requirements & Dependencies

### Python Version

```
Python 3.8 or higher
```

### Required Libraries

```
pandas>=1.3.0              # Data manipulation & analysis
numpy>=1.21.0              # Numerical computing
scikit-learn>=0.24.0       # Machine learning algorithms
seaborn>=0.11.0            # Statistical visualization
matplotlib>=3.4.0          # Plotting & visualization
jupyter>=1.0.0             # Interactive notebooks
ipython>=7.0.0             # Enhanced Python shell
```

---

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/SVLakshman287/Week1.git
cd Week1
```

### Step 2: Create Virtual Environment

**Using venv:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Using conda:**
```bash
conda create -n solar-prediction python=3.9
conda activate solar-prediction
```

### Step 3: Install Dependencies

```bash
pip install pandas numpy scikit-learn seaborn matplotlib jupyter ipython
```

### Step 4: Launch Jupyter Notebook

```bash
jupyter notebook soloarpowerprediction.ipynb
```

The notebook will open at `http://localhost:8888`

---

## 💻 Usage Guide

### 1️⃣ Import Libraries

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

### 2️⃣ Load Dataset

```python
df = pd.read_csv('solarpowergeneration.csv')
print("Dataset Shape:", df.shape)  # (4213, 21)
print(df.head())
print(df.describe())
```

### 3️⃣ Exploratory Data Analysis

```python
# Check for missing values
print(df.isnull().sum())

# Correlation with target
print(df.corr()['generated_power_kw'].sort_values(ascending=False))

# Visualize
sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
plt.show()
```

### 4️⃣ Data Preprocessing

```python
# Separate features and target
X = df.drop('generated_power_kw', axis=1)
y = df['generated_power_kw']

# Split data (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 5️⃣ Train Model

```python
model = LinearRegression()
model.fit(X_train, y_train)
print("✅ Model training completed!")
```

### 6️⃣ Evaluate Model

```python
y_pred_test = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_test)

print(f"MAE:  {mae:.2f} kW")
print(f"RMSE: {rmse:.2f} kW")
print(f"R²:   {r2:.4f}")
```

### 7️⃣ Visualize Results

```python
plt.figure(figsize=(14, 5))

# Actual vs Predicted
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_test, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Power (kW)')
plt.ylabel('Predicted Power (kW)')
plt.title('Actual vs Predicted')
plt.grid(alpha=0.3)

# Residuals
plt.subplot(1, 2, 2)
residuals = y_test - y_pred_test
plt.scatter(y_pred_test, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Power (kW)')
plt.ylabel('Residuals (kW)')
plt.title('Residual Plot')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 📈 Results & Performance

### Key Metrics

| Metric | Description |
|--------|-------------|
| **MAE** | Mean Absolute Error - Average absolute difference |
| **MSE** | Mean Squared Error - Penalizes larger errors |
| **RMSE** | Root Mean Squared Error - In target units |
| **R² Score** | 0-1 scale, higher is better |

### Key Insights

✅ **Solar Geometry** - Angle of incidence, zenith, and azimuth are critical

✅ **Cloud Cover** - Inverse relationship with power generation

✅ **Temperature** - Positive correlation with output

✅ **Seasonal Patterns** - Clear daily cycles in data

✅ **Wind Influence** - Secondary but measurable effect

---

## 🔮 Future Improvements

### Model Enhancements
- [ ] Random Forest Regression
- [ ] Gradient Boosting (XGBoost, LightGBM)
- [ ] Neural Networks (TensorFlow)
- [ ] Hyperparameter Tuning
- [ ] Ensemble Methods

### Feature Engineering
- [ ] Polynomial Features
- [ ] Interaction Terms
- [ ] Temporal Features (hour, day, season)
- [ ] Rolling Averages
- [ ] Derived Features

### Data & Deployment
- [ ] Larger, diverse datasets
- [ ] REST API Creation
- [ ] Cloud Deployment (AWS/Azure/GCP)
- [ ] Real-time Prediction System
- [ ] Web Dashboard

---

## 🎓 Learning Outcomes

✅ Data Science Workflow - From raw data to predictions

✅ Exploratory Data Analysis - Understanding data characteristics

✅ Statistical Analysis - Descriptive statistics & correlations

✅ Machine Learning Fundamentals - Supervised learning

✅ Regression Analysis - Linear relationships modeling

✅ Model Evaluation - Performance metrics interpretation

✅ Python Programming - Pandas, NumPy, Scikit-learn

✅ Data Visualization - Matplotlib & Seaborn

✅ Jupyter Notebooks - Interactive development

---

## 📚 References

### Python & Data Science
- [Python Documentation](https://docs.python.org/3/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

### Visualization
- [Matplotlib Documentation](https://matplotlib.org/)
- [Seaborn Documentation](https://seaborn.pydata.org/)
- [Jupyter Documentation](https://jupyter.org/documentation)

### Machine Learning
- [Linear Regression - Wikipedia](https://en.wikipedia.org/wiki/Linear_regression)
- [ML Course - Andrew Ng](https://www.coursera.org/learn/machine-learning)
- [Hands-On ML Book](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)

### Solar Energy
- [Solar Power - Wikipedia](https://en.wikipedia.org/wiki/Solar_power)
- [Photovoltaics](https://en.wikipedia.org/wiki/Photovoltaics)
- [NREL Solar Data](https://pvwatts.nrel.gov/)

---

## 🤝 Contributing

### How to Contribute

1. **Fork the Repository**
   ```bash
   # Click Fork on GitHub
   ```

2. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature
   ```

3. **Make Changes**
   ```bash
   # Edit files and improve the project
   ```

4. **Commit Changes**
   ```bash
   git commit -m "Add descriptive message"
   ```

5. **Push to Branch**
   ```bash
   git push origin feature/your-feature
   ```

6. **Open Pull Request**
   - Click "New Pull Request" on GitHub
   - Describe your changes

### Contribution Areas

- 🐛 Bug fixes and improvements
- 📚 Better documentation
- 📊 Additional visualizations
- 🤖 New model implementations
- ✨ Feature engineering techniques
- 📝 Tutorial notebooks

---

## 📝 License

MIT License - You are free to use, modify, and distribute this project.

```
MIT License

Copyright (c) 2025 SVLakshman287

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 👤 Author

**SVLakshman287**

- 🐙 GitHub: [@SVLakshman287](https://github.com/SVLakshman287)
- 📦 Repository: [Week1](https://github.com/SVLakshman287/Week1)
- 🎓 Data Science Enthusiast
- 🚀 Machine Learning Practitioner
- 💼 Renewable Energy Advocate

---

## 🎉 Support

If you found this helpful, please:

⭐ **Star this repository** on GitHub

🔖 **Bookmark** for future reference

👥 **Share** with your network

💬 **Provide feedback** for improvements

---

<div align="center">

**Happy Learning! Happy Coding!** 🚀✨

Last Updated: February 2025 | Version: 1.0.0 | Status: ✅ Active

</div>
