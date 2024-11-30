from flask import Flask, request, jsonify
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Definir la ruta donde están los archivos pkl
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
PREPROCESSOR_PATH = os.path.join(BASE_DIR, 'preprocessor.pkl')

# Cargar el preprocesador
with open(PREPROCESSOR_PATH, 'rb') as file:
    preprocessor = pickle.load(file)

# Cargar el modelo
with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return "API de Predicción de Ataque al Corazón"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener datos JSON del request
        data = request.get_json()
        
        # Validar que se haya recibido un JSON
        if not data:
            return jsonify({"error": "No se recibieron datos"}), 400
        
        # Lista de características esperadas
        expected_features = ['age', 'sex', 'trtbps', 'chol', 'thalachh',
                             'oldpeak', 'exng', 'caa', 'cp', 'fbs',
                             'restecg', 'slp', 'thall']
        
        # Verificar que todas las características estén presentes
        missing_features = [feature for feature in expected_features if feature not in data]
        if missing_features:
            return jsonify({"error": f"Faltan las siguientes características: {missing_features}"}), 400
        
        # Crear DataFrame con una sola fila
        input_data = pd.DataFrame([data], columns=expected_features)
        
        # Preprocesar los datos
        X_processed = preprocessor.transform(input_data)
        
        # Realizar la predicción
        prediction = model.predict(X_processed)[0]
        prediction_proba = model.predict_proba(X_processed)[0]
        
        # Preparar la respuesta
        result = {
            "Predicción": "Presencia de ataque al corazón" if prediction == 1 else "Ausencia de ataque al corazón",
            "Probabilidad_Ausencia": round(prediction_proba[0], 2),
            "Probabilidad_Presencia": round(prediction_proba[1], 2)
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

