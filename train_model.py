import os
import pandas as pd
import numpy as np
import joblib
from math import sqrt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==============================
# 1. Cargar datos
# ==============================
train = pd.read_csv("data/train.csv")

# ==============================
# 2. Limpieza de datos
# ==============================
train = train.select_dtypes(include=[np.number])
train = train.dropna(subset=['SalePrice'])
X = train.drop("SalePrice", axis=1)
y = train["SalePrice"]

# ==============================
# 3. Escalado
# ==============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==============================
# 4. Entrenamiento
# ==============================
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20],
    "min_samples_split": [2, 5]
}
rf = RandomForestRegressor(random_state=42)
grid = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, verbose=1)
grid.fit(X_scaled, y)

best_model = grid.best_estimator_
print("Mejores parámetros:", grid.best_params_)

# ==============================
# 5. Evaluación
# ==============================
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

rmse = sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.2f}")

# ==============================
# 6. Guardar modelo
# ==============================
os.makedirs("model", exist_ok=True)
joblib.dump(best_model, "model/house_price_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
print("✅ Modelo y scaler guardados correctamente.")
