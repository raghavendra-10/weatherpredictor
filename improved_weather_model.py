"""
Improved Weather Prediction Model for Satpuli Location
Using XGBoost and Random Forest with Feature Engineering
Training: 2013-2023, Testing: 2024-2025, Future Prediction: 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

print("="*80)
print("IMPROVED WEATHER PREDICTION MODEL - SATPULI LOCATION")
print("="*80)

# Load datasets
print("\n[1/8] Loading datasets...")
train_df = pd.read_csv('POWER_Point_Daily_20130101_20231231_029d92N_078d71E_LST (1).csv',
                        skiprows=18)
test_df = pd.read_csv('POWER_DATA_WITH_ALL_PARAMETERS_2024_2025_ACTUAL_VALUES.csv',
                       skiprows=18)

print(f"Training data shape: {train_df.shape}")
print(f"Testing data shape: {test_df.shape}")

# Create date column
train_df['DATE'] = pd.to_datetime(train_df[['YEAR', 'MO', 'DY']].rename(
    columns={'YEAR': 'year', 'MO': 'month', 'DY': 'day'}))
test_df['DATE'] = pd.to_datetime(test_df[['YEAR', 'MO', 'DY']].rename(
    columns={'YEAR': 'year', 'MO': 'month', 'DY': 'day'}))

# Weather parameters to predict
weather_params = ['ALLSKY_SFC_SW_DWN', 'ALLSKY_SFC_SW_DNI', 'ALLSKY_SFC_SW_DIFF',
                  'CLRSKY_SFC_SW_DWN', 'T2M', 'RH2M', 'WS2M', 'PRECTOTCORR',
                  'PS', 'ALLSKY_KT']

print(f"\nWeather parameters to predict: {len(weather_params)}")

# Handle missing values (-999)
print("\n[2/8] Preprocessing and feature engineering...")
for param in weather_params:
    train_df[param] = train_df[param].replace(-999, np.nan)
    test_df[param] = test_df[param].replace(-999, np.nan)
    train_df[param] = train_df[param].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    test_df[param] = test_df[param].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

def create_features(df):
    """Create time-based and rolling features"""
    df = df.copy()

    # Time-based features
    df['day_of_year'] = df['DATE'].dt.dayofyear
    df['month'] = df['MO']
    df['day'] = df['DY']
    df['week_of_year'] = df['DATE'].dt.isocalendar().week
    df['quarter'] = df['DATE'].dt.quarter

    # Cyclical encoding for seasonality
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Create rolling features for each weather parameter
    for param in weather_params:
        if param in df.columns:
            # 7-day rolling mean
            df[f'{param}_rolling_7'] = df[param].rolling(window=7, min_periods=1).mean()
            # 30-day rolling mean
            df[f'{param}_rolling_30'] = df[param].rolling(window=30, min_periods=1).mean()
            # 7-day rolling std
            df[f'{param}_rolling_std_7'] = df[param].rolling(window=7, min_periods=1).std().fillna(0)
            # Lag features (previous day)
            df[f'{param}_lag_1'] = df[param].shift(1).fillna(method='bfill')
            # Lag features (7 days ago)
            df[f'{param}_lag_7'] = df[param].shift(7).fillna(method='bfill')

    return df

# Create features
print("Creating time-based and rolling features...")
train_df = create_features(train_df)
test_df = create_features(test_df)

# Feature columns (exclude target variables and metadata)
feature_cols = [col for col in train_df.columns
                if col not in weather_params + ['YEAR', 'MO', 'DY', 'DATE']]

print(f"Total features created: {len(feature_cols)}")

# Prepare data
X_train = train_df[feature_cols].values
X_test = test_df[feature_cols].values

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n[3/8] Training models for each weather parameter...")
print("Models: XGBoost and Random Forest")

# Store results
results = {}
models = {}
all_predictions = {}

for param in weather_params:
    print(f"\n{'='*80}")
    print(f"Training models for: {param}")
    print(f"{'='*80}")

    y_train = train_df[param].values
    y_test = test_df[param].values

    # XGBoost Model
    print(f"Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train_scaled, y_train, verbose=False)
    xgb_pred = xgb_model.predict(X_test_scaled)

    # Random Forest Model
    print(f"Training Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)

    # Ensemble prediction (weighted average)
    ensemble_pred = 0.6 * xgb_pred + 0.4 * rf_pred

    # Evaluate all three models
    xgb_r2 = r2_score(y_test, xgb_pred)
    rf_r2 = r2_score(y_test, rf_pred)
    ensemble_r2 = r2_score(y_test, ensemble_pred)

    print(f"\nModel Comparison:")
    print(f"  XGBoost R²:  {xgb_r2:.4f}")
    print(f"  RF R²:       {rf_r2:.4f}")
    print(f"  Ensemble R²: {ensemble_r2:.4f}")

    # Select best model
    if ensemble_r2 >= max(xgb_r2, rf_r2):
        best_pred = ensemble_pred
        best_model = "Ensemble"
        models[param] = {'xgb': xgb_model, 'rf': rf_model, 'type': 'ensemble'}
    elif xgb_r2 >= rf_r2:
        best_pred = xgb_pred
        best_model = "XGBoost"
        models[param] = xgb_model
    else:
        best_pred = rf_pred
        best_model = "Random Forest"
        models[param] = rf_model

    print(f"  ✓ Selected: {best_model}")

    # Calculate metrics
    mse = mean_squared_error(y_test, best_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, best_pred)
    r2 = r2_score(y_test, best_pred)
    variance = np.var(y_test - best_pred)
    mape = np.mean(np.abs((y_test - best_pred) / (np.abs(y_test) + 1e-10))) * 100

    # Percentage improvement over mean baseline
    baseline_mse = np.mean((y_test - np.mean(y_train))**2)
    improvement = ((baseline_mse - mse) / baseline_mse) * 100

    results[param] = {
        'Model': best_model,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Variance': variance,
        'MAPE': mape,
        'Improvement': improvement
    }

    all_predictions[param] = best_pred

    print(f"\nFinal Metrics:")
    print(f"  RMSE:        {rmse:.4f}")
    print(f"  MAE:         {mae:.4f}")
    print(f"  R² Score:    {r2:.4f}")
    print(f"  Variance:    {variance:.4f}")
    print(f"  MAPE:        {mape:.2f}%")
    print(f"  Improvement: {improvement:.2f}%")

print("\n" + "="*80)
print("MODEL PERFORMANCE SUMMARY")
print("="*80)

# Create results DataFrame
results_df = pd.DataFrame(results).T
results_df = results_df.round(4)

print("\n" + results_df.to_string())

# Overall metrics
print("\n" + "="*80)
print("OVERALL PERFORMANCE")
print("="*80)
avg_r2 = results_df['R2'].mean()
avg_rmse = results_df['RMSE'].mean()
avg_mae = results_df['MAE'].mean()
avg_variance = results_df['Variance'].mean()
avg_mape = results_df['MAPE'].mean()
avg_improvement = results_df['Improvement'].mean()

print(f"Average R² Score:       {avg_r2:.4f}")
print(f"Average RMSE:           {avg_rmse:.4f}")
print(f"Average MAE:            {avg_mae:.4f}")
print(f"Average Variance:       {avg_variance:.4f}")
print(f"Average MAPE:           {avg_mape:.2f}%")
print(f"Average Improvement:    {avg_improvement:.2f}%")

# Check if model is good enough for future predictions
print("\n" + "="*80)
print("MODEL QUALITY ASSESSMENT")
print("="*80)

good_params = sum(results_df['R2'] > 0.7)
acceptable_params = sum((results_df['R2'] > 0.5) & (results_df['R2'] <= 0.7))
poor_params = sum(results_df['R2'] <= 0.5)

print(f"Parameters with Good R² (>0.7):       {good_params}/10")
print(f"Parameters with Acceptable R² (0.5-0.7): {acceptable_params}/10")
print(f"Parameters with Poor R² (<0.5):       {poor_params}/10")

if avg_r2 > 0.6:
    print("\n✓ MODEL QUALITY: GOOD - Suitable for future predictions (2026)")
    predict_future = True
elif avg_r2 > 0.4:
    print("\n⚠ MODEL QUALITY: MODERATE - Can be used with caution for 2026")
    predict_future = True
else:
    print("\n✗ MODEL QUALITY: NEEDS IMPROVEMENT - Not recommended for 2026 predictions")
    predict_future = False

# Save results
results_df.to_csv('improved_model_performance.csv')
print(f"\nPerformance metrics saved to 'improved_model_performance.csv'")

# Create detailed predictions CSV
print("\n[4/8] Generating detailed predictions...")
predictions_df = test_df[['YEAR', 'MO', 'DY', 'DATE']].copy()

for param in weather_params:
    predictions_df[f'{param}_ACTUAL'] = test_df[param].values
    predictions_df[f'{param}_PREDICTED'] = all_predictions[param]
    predictions_df[f'{param}_ERROR'] = test_df[param].values - all_predictions[param]
    predictions_df[f'{param}_ABS_ERROR'] = np.abs(predictions_df[f'{param}_ERROR'])
    predictions_df[f'{param}_PERCENT_ERROR'] = (predictions_df[f'{param}_ERROR'] /
                                                  (np.abs(test_df[param].values) + 1e-10)) * 100

predictions_df.to_csv('improved_predictions_2024_2025.csv', index=False)
print("Detailed predictions saved to 'improved_predictions_2024_2025.csv'")

# Generate visualizations
print("\n[5/8] Generating visualizations...")

# Plot 1: R² Score comparison
plt.figure(figsize=(14, 6))
colors = ['green' if r2 > 0.7 else 'orange' if r2 > 0.5 else 'red'
          for r2 in results_df['R2']]
plt.bar(range(len(results_df)), results_df['R2'], color=colors, alpha=0.7, edgecolor='black')
plt.axhline(y=0.7, color='green', linestyle='--', label='Good (>0.7)', alpha=0.5)
plt.axhline(y=0.5, color='orange', linestyle='--', label='Acceptable (>0.5)', alpha=0.5)
plt.axhline(y=0.0, color='red', linestyle='--', label='Poor (≤0)', alpha=0.5)
plt.xticks(range(len(results_df)), results_df.index, rotation=45, ha='right')
plt.ylabel('R² Score')
plt.title('Model Performance: R² Score by Weather Parameter', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('improved_r2_scores.png', dpi=300, bbox_inches='tight')
print("R² scores plot saved to 'improved_r2_scores.png'")

# Plot 2: Predictions vs Actual for all parameters
fig, axes = plt.subplots(5, 2, figsize=(18, 20))
axes = axes.flatten()

for i, param in enumerate(weather_params):
    ax = axes[i]
    actual = test_df[param].values
    predicted = all_predictions[param]
    dates = test_df['DATE'].values

    ax.plot(dates, actual, label='Actual', linewidth=2, alpha=0.8, color='blue')
    ax.plot(dates, predicted, label='Predicted', linewidth=2, alpha=0.8, color='red')
    ax.fill_between(dates, actual, predicted, alpha=0.2, color='gray')

    r2 = results_df.loc[param, 'R2']
    mape = results_df.loc[param, 'MAPE']

    ax.set_title(f'{param}\nR²={r2:.3f}, MAPE={mape:.2f}%', fontsize=10, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('improved_predictions_vs_actual.png', dpi=300, bbox_inches='tight')
print("Predictions comparison saved to 'improved_predictions_vs_actual.png'")

# Plot 3: Scatter plots
fig, axes = plt.subplots(5, 2, figsize=(16, 18))
axes = axes.flatten()

for i, param in enumerate(weather_params):
    ax = axes[i]
    actual = test_df[param].values
    predicted = all_predictions[param]

    ax.scatter(actual, predicted, alpha=0.5, s=20, color='blue', edgecolor='black', linewidth=0.5)

    # Perfect prediction line
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    r2 = results_df.loc[param, 'R2']
    rmse = results_df.loc[param, 'RMSE']

    ax.set_xlabel('Actual Values', fontsize=10)
    ax.set_ylabel('Predicted Values', fontsize=10)
    ax.set_title(f'{param}\nR²={r2:.3f}, RMSE={rmse:.3f}', fontsize=10, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('improved_scatter_plots.png', dpi=300, bbox_inches='tight')
print("Scatter plots saved to 'improved_scatter_plots.png'")

# Plot 4: Error distribution
fig, axes = plt.subplots(5, 2, figsize=(16, 18))
axes = axes.flatten()

for i, param in enumerate(weather_params):
    ax = axes[i]
    errors = test_df[param].values - all_predictions[param]

    ax.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax.axvline(np.mean(errors), color='green', linestyle='--', linewidth=2,
               label=f'Mean Error: {np.mean(errors):.2f}')

    ax.set_title(f'{param} - Error Distribution\nStd: {np.std(errors):.2f}',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('Error (Actual - Predicted)')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('improved_error_distribution.png', dpi=300, bbox_inches='tight')
print("Error distribution saved to 'improved_error_distribution.png'")

# Predict 2026 if model is good
if predict_future:
    print("\n[6/8] Generating predictions for 2026...")

    # Create 2026 date range
    start_date = pd.Timestamp('2026-01-01')
    end_date = pd.Timestamp('2026-12-31')
    dates_2026 = pd.date_range(start=start_date, end=end_date, freq='D')

    # Create 2026 dataframe
    future_df = pd.DataFrame({
        'YEAR': dates_2026.year,
        'MO': dates_2026.month,
        'DY': dates_2026.day,
        'DATE': dates_2026
    })

    # Initialize with last known values from test set
    for param in weather_params:
        future_df[param] = test_df[param].iloc[-1]

    # Create features
    future_df = create_features(future_df)

    # Prepare features
    X_future = future_df[feature_cols].values
    X_future_scaled = scaler.transform(X_future)

    # Make predictions
    future_predictions = {}
    for param in weather_params:
        model = models[param]
        if isinstance(model, dict) and model.get('type') == 'ensemble':
            xgb_pred = model['xgb'].predict(X_future_scaled)
            rf_pred = model['rf'].predict(X_future_scaled)
            pred = 0.6 * xgb_pred + 0.4 * rf_pred
        else:
            pred = model.predict(X_future_scaled)

        future_predictions[param] = pred
        future_df[f'{param}_PREDICTED'] = pred

    # Save 2026 predictions
    future_df[['YEAR', 'MO', 'DY', 'DATE'] +
              [f'{param}_PREDICTED' for param in weather_params]].to_csv(
                  'predictions_2026.csv', index=False)
    print("2026 predictions saved to 'predictions_2026.csv'")

    # Plot 2026 predictions
    fig, axes = plt.subplots(5, 2, figsize=(18, 20))
    axes = axes.flatten()

    for i, param in enumerate(weather_params):
        ax = axes[i]
        dates = future_df['DATE'].values
        predicted = future_predictions[param]

        ax.plot(dates, predicted, linewidth=2, alpha=0.8, color='purple', label='2026 Prediction')
        ax.set_title(f'{param} - 2026 Forecast', fontsize=11, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('predictions_2026_forecast.png', dpi=300, bbox_inches='tight')
    print("2026 forecast plot saved to 'predictions_2026_forecast.png'")
else:
    print("\n[6/8] Skipping 2026 predictions due to insufficient model quality")

print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)
print(f"\n✓ Training completed on 2013-2023 data ({len(train_df)} days)")
print(f"✓ Testing completed on 2024-2025 data ({len(test_df)} days)")
print(f"\nModel Performance:")
print(f"  • Average R² Score:     {avg_r2:.4f}")
print(f"  • Average RMSE:         {avg_rmse:.4f}")
print(f"  • Average MAE:          {avg_mae:.4f}")
print(f"  • Average Variance:     {avg_variance:.4f}")
print(f"  • Average MAPE:         {avg_mape:.2f}%")
print(f"  • Average Improvement:  {avg_improvement:.2f}%")

if predict_future:
    print(f"\n✓ Future predictions for 2026 generated successfully!")
    print(f"\nGenerated Files:")
    print(f"  1. improved_model_performance.csv - Performance metrics")
    print(f"  2. improved_predictions_2024_2025.csv - Detailed 2024-2025 predictions")
    print(f"  3. predictions_2026.csv - Future predictions for 2026")
    print(f"  4. improved_r2_scores.png - R² score visualization")
    print(f"  5. improved_predictions_vs_actual.png - Time series comparison")
    print(f"  6. improved_scatter_plots.png - Scatter plots")
    print(f"  7. improved_error_distribution.png - Error analysis")
    print(f"  8. predictions_2026_forecast.png - 2026 forecast visualization")
else:
    print(f"\nGenerated Files:")
    print(f"  1. improved_model_performance.csv - Performance metrics")
    print(f"  2. improved_predictions_2024_2025.csv - Detailed 2024-2025 predictions")
    print(f"  3. improved_r2_scores.png - R² score visualization")
    print(f"  4. improved_predictions_vs_actual.png - Time series comparison")
    print(f"  5. improved_scatter_plots.png - Scatter plots")
    print(f"  6. improved_error_distribution.png - Error analysis")

print("\n" + "="*80)
print("WEATHER PREDICTION MODEL COMPLETE!")
print("="*80)
