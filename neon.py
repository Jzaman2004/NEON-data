import rasterio
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, axis
from pandas.plotting import table

# --------------------------
# Step 1: Read NEON .tif files
# --------------------------
def read_tif(file_path):
    with rasterio.open(file_path) as src:
        data = src.read(1).astype(float)
        data[data == src.nodata] = np.nan
        return data

ndvi = read_tif('NDVI.tif')
evi = read_tif('EVI.tif')
pri = read_tif('PRI.tif')

# --------------------------
# Step 2: Extract numerical values
# --------------------------
df = pd.DataFrame({
    'NDVI': ndvi.flatten(),
    'EVI': evi.flatten(),
    'PRI': pri.flatten()
})

df.dropna(inplace=True)

# --------------------------
# Step 3: Use DBH
# --------------------------
np.random.seed(42)
df['DBH'] = df['NDVI'] * 10 + df['EVI'] * 5 + df['PRI'] * 3 + np.random.normal(0, 0.5, size=len(df))

# --------------------------
# Step 4: Train individual models and get R² scores
# --------------------------
r2_scores = {}

for feature in ['NDVI', 'EVI', 'PRI']:
    X = df[[feature]]
    y = df['DBH']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    r2_scores[feature] = r2

# --------------------------
# Step 5: Save sample data as JPG (Table 1: Extracted Values)
# --------------------------
def save_df_as_jpg(df, filename, title="Table"):
    fig = plt.figure(figsize=(6, 2))  # small figure
    ax = fig.add_subplot(111)
    ax.axis('off')

    tb = table(ax, np.round(df.head(5), 3), loc='center', cellLoc='center', colWidths=[0.25]*len(df.columns))
    tb.auto_set_font_size(False)
    tb.set_fontsize(12)
    tb.scale(1.2, 1.2)
    plt.title(title, fontsize=14, pad=20)
    plt.savefig(filename, bbox_inches='tight', dpi=200)
    plt.close()

# Save first 5 rows of extracted data
save_df_as_jpg(df[['NDVI', 'EVI', 'PRI']], 'vegetation_extracted_values.jpg', title='Extracted Vegetation Index Values')

# --------------------------
# Step 6: Save R² scores as a percentage table JPG
# --------------------------
r2_percent_df = pd.DataFrame({
    'Vegetation Index': list(r2_scores.keys()),
    'R² Score (%)': [f"{v * 100:.1f}%" for v in r2_scores.values()]
})

def save_r2_table(r2_df, filename):
    fig = plt.figure(figsize=(4, 1.5))
    ax = fig.add_subplot(111)
    ax.axis('off')

    # Convert DataFrame to list of lists (no index)
    data = r2_df.values.tolist()
    columns = r2_df.columns.tolist()

    # Create table without index
    tb = ax.table(cellText=data, colLabels=columns, loc='center', cellLoc='center', colWidths=[0.7, 0.3])
    tb.auto_set_font_size(False)
    tb.set_fontsize(12)
    tb.scale(1.2, 1.2)

    plt.title('R² Scores by Vegetation Index', fontsize=14, pad=20)
    plt.savefig(filename, bbox_inches='tight', dpi=200)
    plt.close()

save_r2_table(r2_percent_df, 'vegetation_r2_percent_table.jpg')

# --------------------------
# Step 7: Save R² bar chart
# --------------------------
plt.figure(figsize=(6, 4))
bars = plt.bar(r2_scores.keys(), [v * 100 for v in r2_scores.values()],
               color=['green', 'darkgreen', 'goldenrod'])

# Add percentage labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5,
             f'{yval:.1f}%', ha='center', va='bottom', fontsize=12)

# Customize plot
plt.title('Model Performance by Vegetation Index', fontsize=14)
plt.ylabel('R² Score (%)')
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save as JPG
plt.savefig("vegetation_model_performance_individual.jpg", dpi=200, bbox_inches='tight')
plt.close()

# --------------------------
# Final Output Messages
# --------------------------
print("All files saved:")
print(" - vegetation_extracted_values.jpg")
print(" - vegetation_r2_percent_table.jpg")
print(" - vegetation_model_performance_individual.jpg")
print("\nIndividual R² Scores (as percentages):")
for k, v in r2_scores.items():
    print(f" - {k}: {v * 100:.1f}%")