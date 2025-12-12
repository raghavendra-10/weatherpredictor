"""
Variance Analysis: Compare Predicted vs Actual 2024-2025
Calculate percentage of records with variance between -0.10 and +0.10
Formula: variance = (actual - predicted) / predicted
Updated to use tighter variance range for better accuracy assessment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("VARIANCE ANALYSIS: PREDICTED VS ACTUAL (2024-2025)")
print("="*80)

# Load the predictions file (using optimized predictions)
print("\nLoading predictions data...")
df = pd.read_csv('optimized_predictions_2024_2025.csv')

# Weather parameters
weather_params = ['ALLSKY_SFC_SW_DWN', 'ALLSKY_SFC_SW_DNI', 'ALLSKY_SFC_SW_DIFF',
                  'CLRSKY_SFC_SW_DWN', 'T2M', 'RH2M', 'WS2M', 'PRECTOTCORR',
                  'PS', 'ALLSKY_KT']

print(f"Total records: {len(df)}")
print(f"Parameters analyzed: {len(weather_params)}")

# Calculate variance for each parameter
print("\n" + "="*80)
print("VARIANCE CALCULATION: (Actual - Predicted) / Predicted")
print("="*80)

results_summary = []

for param in weather_params:
    actual_col = f'{param}_ACTUAL'
    predicted_col = f'{param}_PREDICTED'

    # Get actual and predicted values
    actual = df[actual_col].values
    predicted = df[predicted_col].values

    # Calculate variance: (actual - predicted) / predicted
    # Add small epsilon to avoid division by zero
    variance = (actual - predicted) / (predicted + 1e-10)

    # Store variance in dataframe
    df[f'{param}_VARIANCE'] = variance

    # Count records within range [-0.10, +0.10]
    in_range = ((variance >= -0.10) & (variance <= 0.10)).sum()
    total_records = len(variance)
    percentage = (in_range / total_records) * 100

    # Additional statistics
    mean_variance = np.mean(variance)
    median_variance = np.median(variance)
    std_variance = np.std(variance)
    min_variance = np.min(variance)
    max_variance = np.max(variance)

    # Store results
    results_summary.append({
        'Parameter': param,
        'Total_Records': total_records,
        'Records_In_Range': in_range,
        'Percentage_In_Range': percentage,
        'Mean_Variance': mean_variance,
        'Median_Variance': median_variance,
        'Std_Variance': std_variance,
        'Min_Variance': min_variance,
        'Max_Variance': max_variance
    })

    print(f"\n{param}:")
    print(f"  Total records:           {total_records}")
    print(f"  Records in range:        {in_range}")
    print(f"  Percentage in range:     {percentage:.2f}%")
    print(f"  Mean variance:           {mean_variance:.4f}")
    print(f"  Median variance:         {median_variance:.4f}")
    print(f"  Std deviation:           {std_variance:.4f}")
    print(f"  Min variance:            {min_variance:.4f}")
    print(f"  Max variance:            {max_variance:.4f}")

# Create results dataframe
results_df = pd.DataFrame(results_summary)

# Calculate overall statistics
total_data_points = len(df) * len(weather_params)
total_in_range = results_df['Records_In_Range'].sum()
overall_percentage = (total_in_range / total_data_points) * 100

print("\n" + "="*80)
print("OVERALL SUMMARY")
print("="*80)
print(f"Total data points analyzed:  {total_data_points:,} ({len(df)} days × {len(weather_params)} parameters)")
print(f"Data points in range:        {total_in_range:,}")
print(f"Overall percentage in range: {overall_percentage:.2f}%")

# Save detailed results
results_df.to_csv('variance_analysis_summary.csv', index=False)
print("\nVariance analysis saved to 'variance_analysis_summary.csv'")

# Save detailed variance data
variance_cols = ['YEAR', 'MO', 'DY', 'DATE'] + [f'{param}_VARIANCE' for param in weather_params]
df[variance_cols].to_csv('detailed_variance_data.csv', index=False)
print("Detailed variance data saved to 'detailed_variance_data.csv'")

# ============================================================================
# VARIANCE RANGE BREAKDOWN ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("VARIANCE RANGE BREAKDOWN ANALYSIS")
print("="*80)

# Calculate percentage variance for each parameter
print("\nCalculating percentage variance for detailed breakdown...")

for param in weather_params:
    actual_col = f'{param}_ACTUAL'
    predicted_col = f'{param}_PREDICTED'

    actual = df[actual_col].values
    predicted = df[predicted_col].values

    # Calculate percentage variance: ((actual - predicted) / predicted) * 100
    pct_variance = ((actual - predicted) / (predicted + 1e-10)) * 100
    df[f'{param}_PCT_VARIANCE'] = pct_variance

# Define variance ranges
variance_ranges = [
    ('<-10%', lambda x: x < -10),
    ('<-5%', lambda x: (x >= -10) & (x < -5)),
    ('-5% to 0%', lambda x: (x >= -5) & (x < 0)),
    ('0% to +5%', lambda x: (x >= 0) & (x <= 5)),
    ('>5% to 10%', lambda x: (x > 5) & (x <= 10)),
    ('>10%', lambda x: x > 10)
]

# Create results dataframe for range breakdown
range_results = {}

for param in weather_params:
    pct_variance = df[f'{param}_PCT_VARIANCE'].values

    param_results = []
    for range_name, range_func in variance_ranges:
        count = np.sum(range_func(pct_variance))
        param_results.append(count)

    range_results[param] = param_results

# Add TOTAL column
total_results = []
for i in range(len(variance_ranges)):
    total = sum(range_results[param][i] for param in weather_params)
    total_results.append(total)

range_results['TOTAL'] = total_results

# Create DataFrame
breakdown_df = pd.DataFrame(range_results, index=[r[0] for r in variance_ranges])

# Save to CSV
breakdown_df.to_csv('Actual_vs_Predicted_with_Variance_XGB_10yr.csv')
print("\n✓ Saved: Actual_vs_Predicted_with_Variance_XGB_10yr.csv")

# Display results
print("\n" + "="*80)
print("VARIANCE RANGE BREAKDOWN TABLE")
print("="*80)
print(breakdown_df.to_string())

# Calculate percentages for each range
print("\n" + "="*80)
print("PERCENTAGE DISTRIBUTION BY RANGE")
print("="*80)

total_records = len(df)
total_data_points = total_records * len(weather_params)

for range_name, _ in variance_ranges:
    count = breakdown_df.loc[range_name, 'TOTAL']
    percentage = (count / total_data_points) * 100
    print(f"{range_name:15s}: {count:5d} records ({percentage:5.2f}%)")

# Summary statistics for range breakdown
print("\n" + "="*80)
print("RANGE BREAKDOWN SUMMARY STATISTICS")
print("="*80)

# Good predictions (-5% to +5%)
excellent_predictions = (breakdown_df.loc['-5% to 0%', 'TOTAL'] +
                        breakdown_df.loc['0% to +5%', 'TOTAL'])

# Good predictions (-5% to +10%)
good_predictions = (breakdown_df.loc['-5% to 0%', 'TOTAL'] +
                   breakdown_df.loc['0% to +5%', 'TOTAL'] +
                   breakdown_df.loc['>5% to 10%', 'TOTAL'])

print(f"\nTotal data points:               {total_data_points:,}")
print(f"Excellent (-5% to +5%):          {excellent_predictions:,} ({(excellent_predictions/total_data_points)*100:.2f}%)")
print(f"Good (-5% to +10%):              {good_predictions:,} ({(good_predictions/total_data_points)*100:.2f}%)")
print(f"Outside acceptable range:        {total_data_points - good_predictions:,} ({((total_data_points - good_predictions)/total_data_points)*100:.2f}%)")

# Create visualizations
print("\nGenerating visualizations...")

# Plot 1: Percentage in range by parameter
plt.figure(figsize=(14, 6))
colors = ['green' if p >= 90 else 'orange' if p >= 70 else 'red'
          for p in results_df['Percentage_In_Range']]
bars = plt.bar(range(len(results_df)), results_df['Percentage_In_Range'],
               color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

# Add percentage labels on bars
for i, (bar, pct) in enumerate(zip(bars, results_df['Percentage_In_Range'])):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{pct:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.axhline(y=90, color='green', linestyle='--', label='Excellent (≥90%)', alpha=0.5, linewidth=2)
plt.axhline(y=70, color='orange', linestyle='--', label='Good (≥70%)', alpha=0.5, linewidth=2)
plt.axhline(y=overall_percentage, color='blue', linestyle='-',
            label=f'Overall Average ({overall_percentage:.1f}%)', alpha=0.7, linewidth=2)

plt.xticks(range(len(results_df)), results_df['Parameter'], rotation=45, ha='right')
plt.ylabel('Percentage in Range (%)', fontsize=12, fontweight='bold')
plt.title('Percentage of Records with Variance between -0.10 and +0.10\n(Formula: (Actual - Predicted) / Predicted)',
          fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3, axis='y')
plt.ylim(0, 105)
plt.tight_layout()
plt.savefig('variance_percentage_by_parameter.png', dpi=300, bbox_inches='tight')
print("Saved: variance_percentage_by_parameter.png")

# Plot 2: Variance distribution for each parameter
fig, axes = plt.subplots(5, 2, figsize=(16, 18))
axes = axes.flatten()

for i, param in enumerate(weather_params):
    ax = axes[i]
    variance_data = df[f'{param}_VARIANCE'].values

    # Histogram
    ax.hist(variance_data, bins=50, edgecolor='black', alpha=0.7, color='skyblue')

    # Add vertical lines for range boundaries
    ax.axvline(-0.10, color='red', linestyle='--', linewidth=2, label='Lower bound (-0.10)')
    ax.axvline(0.10, color='red', linestyle='--', linewidth=2, label='Upper bound (+0.10)')
    ax.axvline(0, color='green', linestyle='-', linewidth=2, label='Perfect (0)', alpha=0.7)

    # Shade the acceptable range
    ax.axvspan(-0.10, 0.10, alpha=0.2, color='green', label='Acceptable Range')

    pct = results_df[results_df['Parameter'] == param]['Percentage_In_Range'].values[0]
    mean_var = results_df[results_df['Parameter'] == param]['Mean_Variance'].values[0]

    ax.set_title(f'{param}\n{pct:.1f}% in range | Mean: {mean_var:.3f}',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('Variance: (Actual - Predicted) / Predicted', fontsize=9)
    ax.set_ylabel('Frequency', fontsize=9)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('variance_distribution_histograms.png', dpi=300, bbox_inches='tight')
print("Saved: variance_distribution_histograms.png")

# Plot 3: Variance over time for each parameter
fig, axes = plt.subplots(5, 2, figsize=(18, 20))
axes = axes.flatten()

dates = pd.to_datetime(df['DATE'])

for i, param in enumerate(weather_params):
    ax = axes[i]
    variance_data = df[f'{param}_VARIANCE'].values

    # Plot variance over time
    ax.plot(dates, variance_data, linewidth=1, alpha=0.6, color='blue')

    # Add horizontal lines for range boundaries
    ax.axhline(-0.10, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Lower bound')
    ax.axhline(0.10, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Upper bound')
    ax.axhline(0, color='green', linestyle='-', linewidth=2, alpha=0.5, label='Perfect')

    # Shade the acceptable range
    ax.axhspan(-0.10, 0.10, alpha=0.1, color='green')

    pct = results_df[results_df['Parameter'] == param]['Percentage_In_Range'].values[0]

    ax.set_title(f'{param} - Variance Over Time\n{pct:.1f}% in acceptable range',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('Date', fontsize=9)
    ax.set_ylabel('Variance', fontsize=9)
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('variance_over_time.png', dpi=300, bbox_inches='tight')
print("Saved: variance_over_time.png")

# Plot 4: Summary statistics
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 4a: Mean variance by parameter
ax = axes[0]
colors_mean = ['green' if abs(m) < 0.05 else 'orange' if abs(m) < 0.15 else 'red'
               for m in results_df['Mean_Variance']]
bars = ax.bar(range(len(results_df)), results_df['Mean_Variance'],
              color=colors_mean, alpha=0.7, edgecolor='black', linewidth=1.5)

ax.axhline(0, color='black', linestyle='-', linewidth=2)
ax.set_xticks(range(len(results_df)))
ax.set_xticklabels(results_df['Parameter'], rotation=45, ha='right')
ax.set_ylabel('Mean Variance', fontsize=12, fontweight='bold')
ax.set_title('Mean Variance by Parameter\n(Closer to 0 is better)',
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, results_df['Mean_Variance'])):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height > 0 else -0.01),
            f'{val:.3f}', ha='center', va='bottom' if height > 0 else 'top',
            fontsize=9, fontweight='bold')

# Plot 4b: Standard deviation of variance
ax = axes[1]
colors_std = ['green' if s < 0.2 else 'orange' if s < 0.5 else 'red'
              for s in results_df['Std_Variance']]
bars = ax.bar(range(len(results_df)), results_df['Std_Variance'],
              color=colors_std, alpha=0.7, edgecolor='black', linewidth=1.5)

ax.set_xticks(range(len(results_df)))
ax.set_xticklabels(results_df['Parameter'], rotation=45, ha='right')
ax.set_ylabel('Standard Deviation of Variance', fontsize=12, fontweight='bold')
ax.set_title('Variance Consistency by Parameter\n(Lower is more consistent)',
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, val in zip(bars, results_df['Std_Variance']):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('variance_statistics_summary.png', dpi=300, bbox_inches='tight')
print("Saved: variance_statistics_summary.png")

# Plot 5: Range Breakdown - Stacked Bar Chart
fig, axes = plt.subplots(2, 1, figsize=(16, 12))

# Plot 5a: Count distribution
ax = axes[0]
breakdown_df_plot = breakdown_df.drop('TOTAL', axis=1)

# Define colors for each range
range_colors = ['#d62728', '#ff7f0e', '#ffbb78', '#aec7e8', '#1f77b4', '#2ca02c']
range_names = [r[0] for r in variance_ranges]

# Create stacked bar chart
x_pos = np.arange(len(weather_params))
bottom = np.zeros(len(weather_params))

for i, range_name in enumerate(range_names):
    values = [breakdown_df.loc[range_name, param] for param in weather_params]
    ax.bar(x_pos, values, bottom=bottom, label=range_name,
           color=range_colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)
    bottom += values

ax.set_xticks(x_pos)
ax.set_xticklabels(weather_params, rotation=45, ha='right', fontsize=10)
ax.set_ylabel('Number of Records', fontsize=12, fontweight='bold')
ax.set_title('Variance Range Distribution by Parameter\n(Count of records in each percentage range)',
             fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3, axis='y')

# Add total count on top of each bar
for i, param in enumerate(weather_params):
    total = sum(breakdown_df.loc[range_name, param] for range_name in range_names)
    ax.text(i, total + 10, str(total), ha='center', va='bottom',
            fontsize=9, fontweight='bold')

# Plot 5b: Percentage distribution
ax = axes[1]

# Calculate percentages
percentage_df = breakdown_df_plot.copy()
for param in weather_params:
    total = breakdown_df_plot[param].sum()
    percentage_df[param] = (breakdown_df_plot[param] / total * 100)

# Create stacked bar chart for percentages
x_pos = np.arange(len(weather_params))
bottom = np.zeros(len(weather_params))

for i, range_name in enumerate(range_names):
    values = [percentage_df.loc[range_name, param] for param in weather_params]
    ax.bar(x_pos, values, bottom=bottom, label=range_name,
           color=range_colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)
    bottom += values

ax.set_xticks(x_pos)
ax.set_xticklabels(weather_params, rotation=45, ha='right', fontsize=10)
ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax.set_title('Variance Range Distribution by Parameter\n(Percentage of records in each range)',
             fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig('variance_range_breakdown_stacked.png', dpi=300, bbox_inches='tight')
print("Saved: variance_range_breakdown_stacked.png")

# Plot 6: Heatmap visualization
fig, ax = plt.subplots(figsize=(14, 8))

# Create heatmap data (transpose for better visualization)
heatmap_data = breakdown_df_plot.T

sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='RdYlGn_r',
            linewidths=1, linecolor='black', cbar_kws={'label': 'Count'},
            ax=ax, vmin=0, vmax=300)

ax.set_title('Variance Range Distribution Heatmap\n(Number of records in each percentage range)',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Variance Range', fontsize=12, fontweight='bold')
ax.set_ylabel('Weather Parameter', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('variance_range_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved: variance_range_heatmap.png")

# Create detailed summary table
print("\n" + "="*80)
print("DETAILED RESULTS TABLE")
print("="*80)
print(results_df.to_string(index=False))

print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)
print(f"\n✓ Analysis Complete!")
print(f"\nKey Findings:")
print(f"  • Total records analyzed:     {len(df):,} days")
print(f"  • Total data points:          {total_data_points:,}")
print(f"  • Data points in range:       {total_in_range:,}")
print(f"  • Overall accuracy:           {overall_percentage:.2f}%")
print(f"\nVariance Range: -0.10 to +0.10")
print(f"Formula: (Actual - Predicted) / Predicted")

# Best and worst performing parameters
best_param = results_df.loc[results_df['Percentage_In_Range'].idxmax()]
worst_param = results_df.loc[results_df['Percentage_In_Range'].idxmin()]

print(f"\nBest Performing Parameter:")
print(f"  {best_param['Parameter']}: {best_param['Percentage_In_Range']:.2f}% in range")
print(f"\nWorst Performing Parameter:")
print(f"  {worst_param['Parameter']}: {worst_param['Percentage_In_Range']:.2f}% in range")

print("\n" + "="*80)
print("RANGE BREAKDOWN RESULTS")
print("="*80)
print(f"\nExcellent predictions (-5% to +5%): {(excellent_predictions/total_data_points)*100:.2f}%")
print(f"Good predictions (-5% to +10%):     {(good_predictions/total_data_points)*100:.2f}%")

print("\n" + "="*80)
print("FILES GENERATED")
print("="*80)
print("\nCSV Files:")
print("  1. variance_analysis_summary.csv")
print("  2. detailed_variance_data.csv")
print("  3. Actual_vs_Predicted_with_Variance_XGB_10yr.csv  ← Range breakdown!")
print("\nVisualization Files:")
print("  4. variance_percentage_by_parameter.png")
print("  5. variance_distribution_histograms.png")
print("  6. variance_over_time.png")
print("  7. variance_statistics_summary.png")
print("  8. variance_range_breakdown_stacked.png  ← Range breakdown charts!")
print("  9. variance_range_heatmap.png  ← Range breakdown heatmap!")
print("="*80)
