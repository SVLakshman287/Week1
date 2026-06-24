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
- [❓ FAQ](#-faq)

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

Where:
- 🎯 **Power Output** = Predicted solar power generation
- 📊 **X₁ to X₂₁** = Input features (21 features)
- 📐 **β₀ to β₂₁** = Learned coefficients
- ⚠️ **ε** = Error term

### Model Workflow

### Why Linear Regression?

✅ **Interpretability** - Clear relationship between features and output

✅ **Performance** - Fast training and inference

✅ **Baseline** - Establishes comparison benchmark for advanced models

✅ **Simplicity** - Fewer hyperparameters to tune

✅ **Explainability** - Easy to understand coefficients

---

## 📁 Project Structure

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

### Required Libraries

### Version Compatibility

| Library | Min Version | Recommended |
|---------|-------------|-------------|
| Python | 3.8 | 3.9+ |
| pandas | 1.3.0 | 1.5.0+ |
| numpy | 1.21.0 | 1.23.0+ |
| scikit-learn | 0.24.0 | 1.2.0+ |
| seaborn | 0.11.0 | 0.12.0+ |
| matplotlib | 3.4.0 | 3.6.0+ |

---

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/SVLakshman287/Week1.git
cd Week1
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
# Create conda environment
conda create -n solar-prediction python=3.9

# Activate environment
conda activate solar-prediction
pip install pandas numpy scikit-learn seaborn matplotlib jupyter ipython
# Install all at once
pip install -r requirements.txt
conda install -c conda-forge pandas numpy scikit-learn seaborn matplotlib jupyter
# Start Jupyter server
jupyter notebook soloarpowerprediction.ipynb
