import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

print("1. Loading datasets...")
train_df = pd.read_csv('data/train-test.csv')
val_df = pd.read_csv('data/validation.csv')
december_df = pd.read_csv('data/december-chart-inputs.csv')
template_df = pd.read_csv('data/validation-predictions-template.csv')

# --- DATA PREPARATION FUNCTION ---
def preprocess_data(df, is_training=False, medians=None):
    df_clean = df.copy()
    
    # If we are training, calculate the medians to save for later
    if is_training:
        medians = {
            'market_index': df_clean['market_index'].median(),
            'weight': df_clean['weight'].median(),
            'quote_signal': df_clean['quote_signal'].median(),
            'pickup_lat': df_clean['pickup_lat'].median(),
            'pickup_lon': df_clean['pickup_lon'].median(),
            'delivery_lat': df_clean['delivery_lat'].median(),
            'delivery_lon': df_clean['delivery_lon'].median()
        }
        
    # --- The Trick: Fill in entirely missing columns for the December Dataset ---
    for col, med_val in medians.items():
        if col not in df_clean.columns:
            df_clean[col] = med_val # Fill missing columns with training median

    # 1. Handle Missing Values in existing columns
    df_clean['market_index'] = df_clean['market_index'].fillna(medians['market_index'])
    df_clean['weight'] = df_clean['weight'].fillna(medians['weight'])
    
    # 2. Date Engineering (Extract Month and Day of Week)
    df_clean['date'] = pd.to_datetime(df_clean['date'])
    df_clean['month'] = df_clean['date'].dt.month
    df_clean['day_of_week'] = df_clean['date'].dt.dayofweek
    
    # 3. Categorical Encoding (One-Hot Encode the 'equipment' column)
    df_clean = pd.get_dummies(df_clean, columns=['equipment'], drop_first=True)
    
    # 4. Drop columns we don't need for the math
    cols_to_drop = ['load_id', 'pickup', 'delivery', 'date', 'predicted_rate']
    for col in cols_to_drop:
        if col in df_clean.columns:
            df_clean = df_clean.drop(columns=[col])
    
    # Also drop the target variable if we are returning the training set
    if 'posted_rate' in df_clean.columns:
        y = df_clean['posted_rate']
        X = df_clean.drop(columns=['posted_rate'])
        return X, y, medians
    else:
        return df_clean

print("2. Cleaning Data and Engineering Features...")
X, y, training_medians = preprocess_data(train_df, is_training=True)

# Ensure our validation sets have the exact same columns as our training set
X_val = preprocess_data(val_df, is_training=False, medians=training_medians)
X_dec = preprocess_data(december_df, is_training=False, medians=training_medians)

# Align columns just in case a specific 'equipment' type was missing in the validation sets
X_val = X_val.reindex(columns=X.columns, fill_value=0)
X_dec = X_dec.reindex(columns=X.columns, fill_value=0)

print("3. Splitting Data for Internal Validation (80% Train, 20% Test)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("4. Training the XGBoost Model (This might take 10-15 seconds)...")
model = XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# Evaluate internal performance
y_pred_test = model.predict(X_test)
print("\n=== Model Performance ===")
print(f"Mean Absolute Error (MAE): ${mean_absolute_error(y_test, y_pred_test):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred_test):.4f}")
print("=========================\n")

print("5. Generating Final Predictions...")
# Predict for validation.csv
val_predictions = model.predict(X_val)
template_df['predicted_rate'] = val_predictions
template_df.to_csv('validation_predictions.csv', index=False)
print(" -> Saved 'validation_predictions.csv'")

# Predict for december-chart-inputs.csv
dec_predictions = model.predict(X_dec)
december_df['predicted_rate'] = dec_predictions
december_df.to_csv('data/december-chart-inputs.csv', index=False)
print(" -> Updated 'data/december-chart-inputs.csv'")

print("\n✅ All done! You are ready to run the scorer script.")