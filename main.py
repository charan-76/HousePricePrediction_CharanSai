import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==================================================
# TASK 1 - DATA LOADING & EXPLORATION
# ==================================================

df = pd.read_csv("Housing.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# ==================================================
# TASK 2 - DATA CLEANING
# ==================================================

yes_no_cols = [
    'mainroad',
    'guestroom',
    'basement',
    'hotwaterheating',
    'airconditioning',
    'prefarea'
]

for col in yes_no_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})

df = pd.get_dummies(
    df,
    columns=['furnishingstatus'],
    drop_first=True
)

print("\nShape After Encoding:")
print(df.shape)

# Convert True/False to 1/0 if needed
df = df.astype(int)

# ==================================================
# TASK 3 - MODEL BUILDING
# ==================================================

# Features and Target

X = df.drop('price', axis=1)
y = df['price']

# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTrain Test Split")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

# ==================================================
# LINEAR REGRESSION
# ==================================================

lr_model = LinearRegression()

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

lr_mae = mean_absolute_error(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2 = r2_score(y_test, lr_pred)

print("\nLinear Regression Results")
print("-------------------------")
print("MAE :", round(lr_mae, 2))
print("RMSE:", round(lr_rmse, 2))
print("R² Score:", round(lr_r2, 4))

# ==================================================
# RANDOM FOREST REGRESSOR
# ==================================================

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print("\nRandom Forest Results")
print("----------------------")
print("MAE :", round(rf_mae, 2))
print("RMSE:", round(rf_rmse, 2))
print("R² Score:", round(rf_r2, 4))

# ==================================================
# MODEL COMPARISON
# ==================================================

print("\nModel Comparison")
print("----------------------")

if rf_r2 > lr_r2:
    print("Best Model: Random Forest")
else:
    print("Best Model: Linear Regression")
# ==================================================
# CHART 1 - HOUSE PRICE DISTRIBUTION
# ==================================================

plt.figure(figsize=(8,5))

plt.hist(df['price'], bins=20)

plt.title("House Price Distribution")
plt.xlabel("Price")
plt.ylabel("Number of Houses")

plt.tight_layout()

plt.savefig("charts/price_distribution.png")


# ==================================================
# CHART 2 - CORRELATION HEATMAP
# ==================================================

corr_matrix = df.corr()

plt.figure(figsize=(10,8))

plt.imshow(corr_matrix, aspect='auto')

plt.colorbar()

plt.xticks(
    range(len(corr_matrix.columns)),
    corr_matrix.columns,
    rotation=90
)

plt.yticks(
    range(len(corr_matrix.columns)),
    corr_matrix.columns
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("charts/correlation_heatmap.png")

# ==================================================
# CHART 3 - ACTUAL VS PREDICTED
# ==================================================

plt.figure(figsize=(8,5))

plt.scatter(y_test, lr_pred)

plt.title("Actual vs Predicted House Prices")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")

plt.tight_layout()

plt.savefig("charts/actual_vs_predicted.png")

plt.show()


# ==================================================
# FEATURE IMPORTANCE
# ==================================================

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr_model.coef_
})

feature_importance['Abs_Coefficient'] = abs(feature_importance['Coefficient'])

feature_importance = feature_importance.sort_values(
    by='Abs_Coefficient',
    ascending=False
)

print("\nTop Features Influencing House Price")
print(feature_importance[['Feature', 'Coefficient']].head(10))
