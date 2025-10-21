import os
from flask import Flask, render_template, request
import joblib
import numpy as np

# ---------------------------------------------------------
# Configuración de la aplicación Flask
# ---------------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------------
# Cargar el modelo y el escalador desde rutas relativas
# ---------------------------------------------------------
base_dir = os.path.abspath(os.path.dirname(__file__))
modelo_path = os.path.join(base_dir, '..', 'model', 'house_price_model.pkl')
escalador_path = os.path.join(base_dir, '..', 'model', 'scaler.pkl')

# Convertir rutas a absolutas
modelo_path = os.path.abspath(modelo_path)
escalador_path = os.path.abspath(escalador_path)

# Verificar que existan
if not os.path.exists(modelo_path):
    raise FileNotFoundError(f"❌ No se encontró el modelo en: {modelo_path}")

if not os.path.exists(escalador_path):
    raise FileNotFoundError(f"❌ No se encontró el escalador en: {escalador_path}")

# Cargar los archivos
modelo = joblib.load(modelo_path)
escalador = joblib.load(escalador_path)

# ---------------------------------------------------------
# Página principal
# ---------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

# ---------------------------------------------------------
# Ruta para predecir el precio
# ---------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        area = float(request.form['GrLivArea'])
        calidad = int(request.form['OverallQual'])
        garage = int(request.form['GarageCars'])
        sotano = float(request.form['TotalBsmtSF'])
        anio = int(request.form['YearBuilt'])

        datos = np.array([[area, calidad, garage, sotano, anio]])
        datos_escalados = escalador.transform(datos)
        prediccion = modelo.predict(datos_escalados)[0]

        return render_template('index.html', prediction=round(prediccion, 2))

    except Exception as e:
        return f"❌ Error al realizar la predicción: {e}"

# ---------------------------------------------------------
# Ejecutar la aplicación
# ---------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
