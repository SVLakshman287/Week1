# Solar Power Prediction

A machine learning project that predicts solar power generation output based on meteorological and solar radiation data using Linear Regression.

## Overview

This project analyzes historical solar power generation data and builds a predictive model to forecast future power output. The model considers various environmental factors including temperature, humidity, cloud cover, wind patterns, and solar irradiance to make accurate predictions.

## Dataset

**File:** `solarpowergeneration.csv`

**Size:** 4,213 records with 21 features

**Target Variable:** `generated_power_kw` - Solar power output in kilowatts

### Features

#### Meteorological Data
- `temperature_2_m_above_gnd` - Air temperature at 2m height (°C)
- `relative_humidity_2_m_above_gnd` - Relative humidity at 2m height (%)
- `mean_sea_level_pressure_MSL` - Mean sea level pressure (hPa)
- `total_precipitation_sfc` - Total precipitation (mm)
- `snowfall_amount_sfc` - Snowfall amount (mm)

#### Cloud Cover Data
- `total_cloud_cover_sfc` - Total cloud cover (%)
- `high_cloud_cover_high_cld_lay` - High cloud cover (%)
- `medium_cloud_cover_mid_cld_lay` - Medium cloud cover (%)
- `low_cloud_cover_low_cld_lay` - Low cloud cover (%)

#### Solar Radiation Data
- `shortwave_radiation_backwards_sfc` - Downward shortwave radiation (W/m²)

#### Wind Data
- `wind_speed_10_m_above_gnd` - Wind speed at 10m height (m/s)
- `wind_direction_10_m_above_gnd` - Wind direction at 10m height (°)
- `wind_speed_80_m_above_gnd` - Wind speed at 80m height (m/s)
- `wind_direction_80_m_above_gnd` - Wind direction at 80m height (°)
- `wind_speed_900_mb` - Wind speed at 900 mb pressure level (m/s)
- `wind_direction_900_mb` - Wind direction at 900 mb pressure level (°)
- `wind_gust_10_m_above_gnd` - Wind gust speed at 10m height (m/s)

#### Solar Geometry Data
- `angle_of_incidence` - Solar angle relative to panel surface (°)
- `zenith` - Solar zenith angle (°)
- `azimuth` - Solar azimuth angle (°)

### Data Statistics

| Metric | Value |
|--------|-------|
| Total Records | 4,213 |
| Temperature Range | -5.35°C to 34.9°C |
| Power Output Range | 0.0006 kW to 3,056.79 kW |
| Average Power Output | 1,134.35 kW |

## Model

**Algorithm:** Linear Regression

The model uses scikit-learn's LinearRegression to establish linear relationships between meteorological/solar features and power output.

## Files

| File | Description |
|------|-------------|
| `soloarpowerprediction.ipynb` | Main Jupyter notebook with data analysis and model training |
| `solarpowergeneration.csv` | Historical solar power generation dataset |
| `Solar_Power_Prediction_Presentation.pptx` | Presentation slides with findings and results |
| `README.md` | This file |

## Requirements
pandas>=1.3.0 numpy>=1.21.0 scikit-learn>=0.24.0 seaborn>=0.11.0 matplotlib>=3.4.0 jupyter>=1.0.0

## Installation

1. Clone the repository:
```bash
git clone https://github.com/SVLakshman287/Week1.git
cd Week1
