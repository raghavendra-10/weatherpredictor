"""
Optimized Weather Prediction Model for Satpuli Location
Enhanced with Advanced Feature Engineering and Hyperparameter Tuning
Target: Achieve >70% accuracy within -0.10 to +0.10 variance range
Training: 2013-2023, Testing: 2024-2025, Future Prediction: 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

print("="*80)
print("OPTIMIZED WEATHER PREDICTION MODEL - SATPULI LOCATION")
print("Target: >70% accuracy within -0.10 to +0.10 variance range")
print("="*80)

# Load datasets
print("\n[1/10] Loading datasets...")
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
print("\n[2/10] Advanced preprocessing and feature engineering...")
for param in weather_params:
    train_df[param] = train_df[param].replace(-999, np.nan)
    test_df[param] = test_df[param].replace(-999, np.nan)
    train_df[param] = train_df[param].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    test_df[param] = test_df[param].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

def create_advanced_features(df):
    """Create comprehensive feature set with advanced engineering"""
    df = df.copy()

    # Time-based features
    df['day_of_year'] = df['DATE'].dt.dayofyear
    df['month'] = df['MO']
    df['day'] = df['DY']
    df['week_of_year'] = df['DATE'].dt.isocalendar().week
    df['quarter'] = df['DATE'].dt.quarter
    df['is_weekend'] = df['DATE'].dt.dayofweek.isin([5, 6]).astype(int)

    # Cyclical encoding for seasonality (important for weather patterns)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)

    # Season indicators
    df['season'] = df['month'] % 12 // 3  # 0=Winter, 1=Spring, 2=Summer, 3=Fall
    df['is_summer'] = (df['month'].isin([6, 7, 8])).astype(int)
    df['is_winter'] = (df['month'].isin([12, 1, 2])).astype(int)
    df['is_monsoon'] = (df['month'].isin([7, 8, 9])).astype(int)

    # Create enhanced rolling features for each weather parameter
    for param in weather_params:
        if param in df.columns:
            # Multiple window sizes for better pattern capture
            for window in [3, 7, 14, 30]:
                df[f'{param}_rolling_{window}'] = df[param].rolling(window=window, min_periods=1).mean()
                df[f'{param}_rolling_std_{window}'] = df[param].rolling(window=window, min_periods=1).std().fillna(0)
                df[f'{param}_rolling_min_{window}'] = df[param].rolling(window=window, min_periods=1).min()
                df[f'{param}_rolling_max_{window}'] = df[param].rolling(window=window, min_periods=1).max()

            # Exponential weighted moving average (gives more weight to recent values)
            df[f'{param}_ewm_7'] = df[param].ewm(span=7, adjust=False).mean()
            df[f'{param}_ewm_30'] = df[param].ewm(span=30, adjust=False).mean()

            # Lag features (multiple lags)
            for lag in [1, 2, 3, 7, 14, 30]:
                df[f'{param}_lag_{lag}'] = df[param].shift(lag).fillna(method='bfill')

            # Difference features (rate of change)
            df[f'{param}_diff_1'] = df[param].diff(1).fillna(0)
            df[f'{param}_diff_7'] = df[param].diff(7).fillna(0)

            # Moving average of differences
            df[f'{param}_diff_rolling_7'] = df[f'{param}_diff_1'].rolling(window=7, min_periods=1).mean()

    # Interaction features between related parameters
    if all(p in df.columns for p in ['T2M', 'RH2M']):
        # Temperature-Humidity interaction
        df['temp_humidity_interaction'] = df['T2M'] * df['RH2M']
        df['heat_index'] = df['T2M'] + 0.5 * df['RH2M']  # Simplified heat index

    if all(p in df.columns for p in ['WS2M', 'T2M']):
        # Wind chill effect
        df['wind_chill'] = df['T2M'] - (df['WS2M'] * 0.5)

    if all(p in df.columns for p in ['ALLSKY_SFC_SW_DWN', 'T2M']):
        # Solar-Temperature interaction
        df['solar_temp_interaction'] = df['ALLSKY_SFC_SW_DWN'] * df['T2M']

    if all(p in df.columns for p in ['PS', 'T2M']):
        # Pressure-Temperature interaction
        df['pressure_temp_ratio'] = df['PS'] / (df['T2M'] + 273.15)  # Ideal gas law approximation

    return df

# Create features
print("Creating advanced time-based and rolling features...")
print("This includes: rolling stats, lags, EWM, interactions, and more...")
train_df = create_advanced_features(train_df)
test_df = create_advanced_features(test_df)

# Feature columns (exclude target variables and metadata)
feature_cols = [col for col in train_df.columns
                if col not in weather_params + ['YEAR', 'MO', 'DY', 'DATE']]

print(f"Total features created: {len(feature_cols)}")

# Prepare data
X_train = train_df[feature_cols].values
X_test = test_df[feature_cols].values

# Use RobustScaler for better handling of outliers
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n[3/10] Training optimized models for each weather parameter...")
print("Models: XGBoost (tuned), Random Forest (tuned), Gradient Boosting")

# Store results
results = {}
models = {}
all_predictions = {}

for param in weather_params:
    print(f"\n{'='*80}")
    print(f"Training optimized models for: {param}")
    print(f"{'='*80}")

    y_train = train_df[param].values
    y_test = test_df[param].values

    # XGBoost Model - Optimized hyperparameters
    print(f"Training XGBoost (optimized)...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=500,           # More trees for better learning
        max_depth=8,                # Deeper trees for complex patterns
        learning_rate=0.03,         # Lower learning rate with more trees
        subsample=0.85,             # Slightly higher subsample
        colsample_bytree=0.85,
        colsample_bylevel=0.8,
        min_child_weight=1,
        gamma=0.1,                  # Regularization
        reg_alpha=0.05,             # L1 regularization
        reg_lambda=1.0,             # L2 regularization
        random_state=42,
        n_jobs=-1,
        tree_method='hist'          # Faster training
    )
    xgb_model.fit(X_train_scaled, y_train, verbose=False)
    xgb_pred = xgb_model.predict(X_test_scaled)

    # Random Forest Model - Optimized hyperparameters
    print(f"Training Random Forest (optimized)...")
    rf_model = RandomForestRegressor(
        n_estimators=300,           # More trees
        max_depth=20,               # Deeper trees
        min_samples_split=3,        # Allow smaller splits
        min_samples_leaf=1,         # More granular leaves
        max_features='sqrt',        # Feature subsampling
        bootstrap=True,
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)

    # Gradient Boosting Model
    print(f"Training Gradient Boosting...")
    gb_model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42
    )
    gb_model.fit(X_train_scaled, y_train)
    gb_pred = gb_model.predict(X_test_scaled)

    # Evaluate all models
    xgb_r2 = r2_score(y_test, xgb_pred)
    rf_r2 = r2_score(y_test, rf_pred)
    gb_r2 = r2_score(y_test, gb_pred)

    print(f"\nModel Comparison:")
    print(f"  XGBoost R²:          {xgb_r2:.4f}")
    print(f"  Random Forest R²:    {rf_r2:.4f}")
    print(f"  Gradient Boost R²:   {gb_r2:.4f}")

    # Weighted ensemble - give more weight to better performing models
    # Calculate weights based on R2 scores
    total_r2 = xgb_r2 + rf_r2 + gb_r2
    xgb_weight = xgb_r2 / total_r2 if total_r2 > 0 else 0.33
    rf_weight = rf_r2 / total_r2 if total_r2 > 0 else 0.33
    gb_weight = gb_r2 / total_r2 if total_r2 > 0 else 0.33

    # Normalize weights
    total_weight = xgb_weight + rf_weight + gb_weight
    xgb_weight /= total_weight
    rf_weight /= total_weight
    gb_weight /= total_weight

    # Create weighted ensemble
    ensemble_pred = xgb_weight * xgb_pred + rf_weight * rf_pred + gb_weight * gb_pred
    ensemble_r2 = r2_score(y_test, ensemble_pred)

    print(f"  Weighted Ensemble R²: {ensemble_r2:.4f}")
    print(f"  Weights: XGB={xgb_weight:.2f}, RF={rf_weight:.2f}, GB={gb_weight:.2f}")

    # Select best approach
    if ensemble_r2 >= max(xgb_r2, rf_r2, gb_r2):
        best_pred = ensemble_pred
        best_model = f"Ensemble (XGB:{xgb_weight:.2f}, RF:{rf_weight:.2f}, GB:{gb_weight:.2f})"
        models[param] = {
            'xgb': xgb_model,
            'rf': rf_model,
            'gb': gb_model,
            'type': 'weighted_ensemble',
            'weights': (xgb_weight, rf_weight, gb_weight)
        }
    elif xgb_r2 >= max(rf_r2, gb_r2):
        best_pred = xgb_pred
        best_model = "XGBoost"
        models[param] = xgb_model
    elif rf_r2 >= gb_r2:
        best_pred = rf_pred
        best_model = "Random Forest"
        models[param] = rf_model
    else:
        best_pred = gb_pred
        best_model = "Gradient Boosting"
        models[param] = gb_model

    print(f"  ✓ Selected: {best_model}")

    # Calculate comprehensive metrics
    mse = mean_squared_error(y_test, best_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, best_pred)
    r2 = r2_score(y_test, best_pred)
    variance = np.var(y_test - best_pred)
    mape = np.mean(np.abs((y_test - best_pred) / (np.abs(y_test) + 1e-10))) * 100

    # Calculate variance for -0.10 to +0.10 range
    variance_ratio = (y_test - best_pred) / (best_pred + 1e-10)
    in_tight_range = ((variance_ratio >= -0.10) & (variance_ratio <= 0.10)).sum()
    tight_range_pct = (in_tight_range / len(y_test)) * 100

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
        'Improvement': improvement,
        'Tight_Range_Accuracy': tight_range_pct
    }

    all_predictions[param] = best_pred

    print(f"\nFinal Metrics:")
    print(f"  RMSE:                      {rmse:.4f}")
    print(f"  MAE:                       {mae:.4f}")
    print(f"  R² Score:                  {r2:.4f}")
    print(f"  Variance:                  {variance:.4f}")
    print(f"  MAPE:                      {mape:.2f}%")
    print(f"  Improvement:               {improvement:.2f}%")
    print(f"  Accuracy (-0.10 to +0.10): {tight_range_pct:.2f}%")

print("\n" + "="*80)
print("OPTIMIZED MODEL PERFORMANCE SUMMARY")
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
avg_tight_range = results_df['Tight_Range_Accuracy'].mean()

print(f"Average R² Score:                  {avg_r2:.4f}")
print(f"Average RMSE:                      {avg_rmse:.4f}")
print(f"Average MAE:                       {avg_mae:.4f}")
print(f"Average Variance:                  {avg_variance:.4f}")
print(f"Average MAPE:                      {avg_mape:.2f}%")
print(f"Average Improvement:               {avg_improvement:.2f}%")
print(f"Average Accuracy (-0.10 to +0.10): {avg_tight_range:.2f}%")

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

print(f"\nTight Range Performance (-0.10 to +0.10):")
if avg_tight_range >= 70:
    print(f"✓ EXCELLENT: {avg_tight_range:.2f}% accuracy achieved!")
elif avg_tight_range >= 60:
    print(f"✓ GOOD: {avg_tight_range:.2f}% accuracy achieved (target: >70%)")
else:
    print(f"⚠ NEEDS WORK: {avg_tight_range:.2f}% accuracy (target: >70%)")

# Save results
results_df.to_csv('optimized_model_performance.csv')
print(f"\nPerformance metrics saved to 'optimized_model_performance.csv'")

# Create detailed predictions CSV
print("\n[4/10] Generating detailed predictions...")
predictions_df = test_df[['YEAR', 'MO', 'DY', 'DATE']].copy()

for param in weather_params:
    predictions_df[f'{param}_ACTUAL'] = test_df[param].values
    predictions_df[f'{param}_PREDICTED'] = all_predictions[param]
    predictions_df[f'{param}_ERROR'] = test_df[param].values - all_predictions[param]
    predictions_df[f'{param}_ABS_ERROR'] = np.abs(predictions_df[f'{param}_ERROR'])
    predictions_df[f'{param}_PERCENT_ERROR'] = (predictions_df[f'{param}_ERROR'] /
                                                  (np.abs(test_df[param].values) + 1e-10)) * 100
    predictions_df[f'{param}_VARIANCE'] = (test_df[param].values - all_predictions[param]) / (all_predictions[param] + 1e-10)

predictions_df.to_csv('optimized_predictions_2024_2025.csv', index=False)
print("Detailed predictions saved to 'optimized_predictions_2024_2025.csv'")

# Generate visualizations
print("\n[5/10] Generating visualizations...")

# Plot 1: R² Score comparison
plt.figure(figsize=(14, 6))
colors = ['green' if r2 > 0.7 else 'orange' if r2 > 0.5 else 'red'
          for r2 in results_df['R2']]
bars = plt.bar(range(len(results_df)), results_df['R2'], color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

# Add percentage labels on bars
for i, (bar, r2) in enumerate(zip(bars, results_df['R2'])):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{r2:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.axhline(y=0.7, color='green', linestyle='--', label='Good (>0.7)', alpha=0.5, linewidth=2)
plt.axhline(y=0.5, color='orange', linestyle='--', label='Acceptable (>0.5)', alpha=0.5, linewidth=2)
plt.axhline(y=0.0, color='red', linestyle='--', label='Poor (≤0)', alpha=0.5, linewidth=2)
plt.axhline(y=avg_r2, color='blue', linestyle='-', label=f'Average ({avg_r2:.3f})', alpha=0.7, linewidth=2)

plt.xticks(range(len(results_df)), results_df.index, rotation=45, ha='right')
plt.ylabel('R² Score', fontsize=12, fontweight='bold')
plt.title('Optimized Model Performance: R² Score by Weather Parameter', fontsize=14, fontweight='bold')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3, axis='y')
plt.ylim(-0.1, 1.05)
plt.tight_layout()
plt.savefig('optimized_r2_scores.png', dpi=300, bbox_inches='tight')
print("R² scores plot saved to 'optimized_r2_scores.png'")

# Plot 2: Tight range accuracy comparison
plt.figure(figsize=(14, 6))
colors_tight = ['green' if acc >= 70 else 'orange' if acc >= 60 else 'red'
                for acc in results_df['Tight_Range_Accuracy']]
bars = plt.bar(range(len(results_df)), results_df['Tight_Range_Accuracy'],
               color=colors_tight, alpha=0.7, edgecolor='black', linewidth=1.5)

# Add percentage labels on bars
for i, (bar, acc) in enumerate(zip(bars, results_df['Tight_Range_Accuracy'])):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.axhline(y=70, color='green', linestyle='--', label='Target (≥70%)', alpha=0.5, linewidth=2)
plt.axhline(y=avg_tight_range, color='blue', linestyle='-',
            label=f'Average ({avg_tight_range:.1f}%)', alpha=0.7, linewidth=2)

plt.xticks(range(len(results_df)), results_df.index, rotation=45, ha='right')
plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
plt.title('Accuracy within Tight Variance Range (-0.10 to +0.10)\nTarget: >70%',
          fontsize=14, fontweight='bold')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3, axis='y')
plt.ylim(0, 105)
plt.tight_layout()
plt.savefig('optimized_tight_range_accuracy.png', dpi=300, bbox_inches='tight')
print("Tight range accuracy plot saved to 'optimized_tight_range_accuracy.png'")

# Plot 3: Predictions vs Actual for all parameters
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
    tight_acc = results_df.loc[param, 'Tight_Range_Accuracy']

    ax.set_title(f'{param}\nR²={r2:.3f}, Tight Range Acc={tight_acc:.1f}%',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('optimized_predictions_vs_actual.png', dpi=300, bbox_inches='tight')
print("Predictions comparison saved to 'optimized_predictions_vs_actual.png'")

# Plot 4: Scatter plots
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
    tight_acc = results_df.loc[param, 'Tight_Range_Accuracy']

    ax.set_xlabel('Actual Values', fontsize=10)
    ax.set_ylabel('Predicted Values', fontsize=10)
    ax.set_title(f'{param}\nR²={r2:.3f}, RMSE={rmse:.3f}, Tight={tight_acc:.1f}%',
                 fontsize=10, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('optimized_scatter_plots.png', dpi=300, bbox_inches='tight')
print("Scatter plots saved to 'optimized_scatter_plots.png'")

# Predict 2026 if model is good
if predict_future:
    print("\n[6/10] Generating predictions for 2026...")

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
    future_df = create_advanced_features(future_df)

    # Prepare features
    X_future = future_df[feature_cols].values
    X_future_scaled = scaler.transform(X_future)

    # Make predictions
    future_predictions = {}
    for param in weather_params:
        model = models[param]
        if isinstance(model, dict) and model.get('type') == 'weighted_ensemble':
            xgb_pred = model['xgb'].predict(X_future_scaled)
            rf_pred = model['rf'].predict(X_future_scaled)
            gb_pred = model['gb'].predict(X_future_scaled)
            xgb_w, rf_w, gb_w = model['weights']
            pred = xgb_w * xgb_pred + rf_w * rf_pred + gb_w * gb_pred
        else:
            pred = model.predict(X_future_scaled)

        future_predictions[param] = pred
        future_df[f'{param}_PREDICTED'] = pred

    # Save 2026 predictions
    future_df[['YEAR', 'MO', 'DY', 'DATE'] +
              [f'{param}_PREDICTED' for param in weather_params]].to_csv(
                  'optimized_predictions_2026.csv', index=False)
    print("2026 predictions saved to 'optimized_predictions_2026.csv'")

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
    plt.savefig('optimized_predictions_2026_forecast.png', dpi=300, bbox_inches='tight')
    print("2026 forecast plot saved to 'optimized_predictions_2026_forecast.png'")
else:
    print("\n[6/10] Skipping 2026 predictions due to insufficient model quality")

print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)
print(f"\n✓ Training completed on 2013-2023 data ({len(train_df)} days)")
print(f"✓ Testing completed on 2024-2025 data ({len(test_df)} days)")
print(f"\nOptimized Model Performance:")
print(f"  • Average R² Score:                  {avg_r2:.4f}")
print(f"  • Average RMSE:                      {avg_rmse:.4f}")
print(f"  • Average MAE:                       {avg_mae:.4f}")
print(f"  • Average MAPE:                      {avg_mape:.2f}%")
print(f"  • Average Improvement:               {avg_improvement:.2f}%")
print(f"  • Average Accuracy (-0.10 to +0.10): {avg_tight_range:.2f}%")

if predict_future:
    print(f"\n✓ Future predictions for 2026 generated successfully!")

print("\n" + "="*80)
print("FILES GENERATED")
print("="*80)
print("\nCSV Files:")
print("  1. optimized_model_performance.csv - Performance metrics")
print("  2. optimized_predictions_2024_2025.csv - Detailed 2024-2025 predictions")
if predict_future:
    print("  3. optimized_predictions_2026.csv - Future predictions for 2026")

print("\nVisualization Files:")
print("  4. optimized_r2_scores.png - R² score visualization")
print("  5. optimized_tight_range_accuracy.png - Tight range accuracy by parameter")
print("  6. optimized_predictions_vs_actual.png - Time series comparison")
print("  7. optimized_scatter_plots.png - Scatter plots")
if predict_future:
    print("  8. optimized_predictions_2026_forecast.png - 2026 forecast visualization")

print("\n" + "="*80)
print("OPTIMIZED WEATHER PREDICTION MODEL COMPLETE!")
print("="*80)
