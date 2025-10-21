import os
from flask import Flask, render_template, request
import joblib
import numpy as np

# ---------------------------------------------------------
# Configuración de la aplicación Flask
# ---------------------------------------------------------
app = Flask(__name__)

# Ruta al modelo y escalador (ajusta si están en otra carpeta)
modelo_path = r"C:\Users\Usuario\Desktop\house_price_ml\model\house_price_model.pkl"
escalador_path = r"C:\Users\Usuario\Desktop\house_price_ml\model\scaler.pkl"

# Cargar el modelo y el escalador
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
        # Obtener los valores del formulario
        area = float(request.form['GrLivArea'])
        calidad = int(request.form['OverallQual'])
        garage = int(request.form['GarageCars'])
        sotano = float(request.form['TotalBsmtSF'])
        anio = int(request.form['YearBuilt'])

        # Crear un arreglo con los datos ingresados
        datos = np.array([[area, calidad, garage, sotano, anio]])

        # Escalar los datos
        datos_escalados = escalador.transform(datos)

        # Hacer la predicción
        prediccion = modelo.predict(datos_escalados)[0]

        # Enviar el resultado a la plantilla
        return render_template('index.html', prediction=prediccion)

    except Exception as e:
        return f"❌ Error al realizar la predicción: {e}"

# ---------------------------------------------------------
# Ejecutar la aplicación
# ---------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
