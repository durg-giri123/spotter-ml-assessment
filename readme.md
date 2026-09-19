# Freight Rate Prediction Challenge

## Overview
This repository contains the solution for the Spotter Machine Learning Engineer assessment. The model predicts freight rates using an XGBoost Regressor, achieving an R2 score of 0.85 and an MAE of ~$142 on the internal validation set.

## Project Structure
- `/notebook/eda.py` - Exploratory Data Analysis script.
- `/notebook/train_model.py` - Main pipeline for data cleaning, feature engineering, model training, and prediction generation.
- `validation_predictions.csv` - The final output predictions.

## Setup & Run Instructions
1. Ensure you have Python 3.8+ installed.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Run the training and prediction pipeline:

```bash
python notebook/train_model.py
```

4. Run the scoring script:

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december-chart-inputs.csv
```s