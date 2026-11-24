from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "pipeline_diabetes.joblib"
METRICS_PATH = BASE_DIR / "metricas_modelo.json"
KNOWN_CASE_PATH = BASE_DIR / "caso_prueba_conocido.json"

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

FORM_RANGES = {
    "Pregnancies": (0, 20),
    "Glucose": (1, 300),
    "BloodPressure": (1, 200),
    "SkinThickness": (0, 100),
    "Insulin": (0, 900),
    "BMI": (1, 80),
    "DiabetesPedigreeFunction": (0, 3),
    "Age": (1, 120)
}

st.set_page_config(
    page_title="Predicción académica de diabetes",
    page_icon="🩺",
    layout="centered"
)

@st.cache_resource
def load_model():
    # Cargar únicamente el archivo .joblib generado por el notebook del proyecto.
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_json(path_string):
    path = Path(path_string)
    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    return None

st.title("🩺 Clasificación académica de diabetes")
st.write(
    "Aplicación del proyecto de Minería de Datos. El modelo seleccionado "
    "es el árbol de decisión entrenado en el notebook final."
)
st.info(
    "Herramienta con fines exclusivamente académicos. No realiza diagnósticos "
    "médicos ni sustituye la valoración de un profesional de salud."
)

if not MODEL_PATH.exists():
    st.error(
        "No se encontró `pipeline_diabetes.joblib`. Ejecuta todas las celdas "
        "del notebook final y copia ese archivo a la misma carpeta que `app.py`."
    )
    st.stop()

try:
    model = load_model()
except Exception as exc:
    st.error(
        "No fue posible cargar el modelo. Revisa que `requirements.txt` conserve "
        "las versiones utilizadas al entrenarlo y que el archivo .joblib sea el "
        "generado por el notebook final."
    )
    st.exception(exc)
    st.stop()

metrics = load_json(str(METRICS_PATH))
if metrics:
    st.caption(
        f"Modelo registrado: {metrics.get('selected_model', 'No especificado')} · "
        f"Registros del conjunto: {metrics.get('dataset_rows', 'No especificado')}"
    )

st.subheader("Ingrese los datos")
st.write("Complete las ocho variables utilizadas por el modelo.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Embarazos", min_value=0, max_value=20, value=1, step=1)
        glucose = st.number_input("Glucosa", min_value=1.0, max_value=300.0, value=120.0)
        blood_pressure = st.number_input("Presión arterial", min_value=1.0, max_value=200.0, value=70.0)
        skin_thickness = st.number_input("Grosor de piel", min_value=0.0, max_value=100.0, value=20.0)

    with col2:
        insulin = st.number_input("Insulina", min_value=0.0, max_value=900.0, value=80.0)
        bmi = st.number_input("BMI / IMC", min_value=1.0, max_value=80.0, value=30.0)
        dpf = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0, max_value=3.0, value=0.5, format="%.3f"
        )
        age = st.number_input("Edad", min_value=1, max_value=120, value=30, step=1)

    submitted = st.form_submit_button("Evaluar registro", type="primary", use_container_width=True)

if submitted:
    record = {
        "Pregnancies": float(pregnancies),
        "Glucose": float(glucose),
        "BloodPressure": float(blood_pressure),
        "SkinThickness": float(skin_thickness),
        "Insulin": float(insulin),
        "BMI": float(bmi),
        "DiabetesPedigreeFunction": float(dpf),
        "Age": float(age)
    }

    errors = []
    for feature in FEATURES:
        low, high = FORM_RANGES[feature]
        value = record[feature]
        if value < low or value > high:
            errors.append(f"{feature}: el valor debe estar entre {low} y {high}.")

    if errors:
        for error in errors:
            st.error(error)
    else:
        input_df = pd.DataFrame([record], columns=FEATURES)
        prediction = int(model.predict(input_df)[0])

        st.divider()
        st.subheader("Resultado del modelo")
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            positive_probability = float(probabilities[1])
            st.metric("Probabilidad estimada para la clase 1", f"{positive_probability:.1%}")
            st.progress(max(0.0, min(1.0, positive_probability)))

        if prediction == 1:
            st.warning(
                "El modelo clasificó este registro en la clase 1 (resultado positivo). "
                "Esto no significa que la persona tenga diabetes."
            )
        else:
            st.success(
                "El modelo clasificó este registro en la clase 0 (resultado negativo). "
                "Esto no descarta una condición médica."
            )

st.divider()
st.subheader("Información del proyecto")
st.write("**Integrantes:** José Rivas, Sebastián Muñoz y Camilo Camargo.")
st.write("**Modelos comparados:** regresión logística y árbol de decisión.")
st.write("**Modelo seleccionado:** árbol de decisión, priorizando el recall de la clase 1 en validación cruzada.")

if metrics and "models" in metrics:
    with st.expander("Consultar métricas registradas en el notebook"):
        rows = []
        for model_name, values in metrics["models"].items():
            rows.append({
                "Modelo": model_name,
                "Accuracy prueba": values.get("accuracy"),
                "Precision clase 1": values.get("precision_1"),
                "Recall clase 1": values.get("recall_1"),
                "F1 clase 1": values.get("f1_1"),
                "AUC prueba": values.get("roc_auc"),
                "Recall CV": values.get("cv_recall_mean")
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with st.expander("Verificación del modelo serializado"):
    known_case = load_json(str(KNOWN_CASE_PATH))
    if known_case:
        st.write("El notebook guardó un caso de prueba para comprobar la consistencia del archivo serializado.")
        if st.button("Ejecutar comprobación del caso conocido"):
            test_df = pd.DataFrame([known_case["features"]], columns=FEATURES)
            app_prediction = int(model.predict(test_df)[0])
            expected = int(known_case["expected_prediction"])
            if app_prediction == expected:
                st.success(f"Prueba superada: predicción de la app = {app_prediction}, predicción esperada = {expected}.")
            else:
                st.error(f"Las predicciones no coinciden: app = {app_prediction}, esperada = {expected}. Revisa el modelo.")
    else:
        st.caption("Copia `caso_prueba_conocido.json` desde la ejecución del notebook para habilitar esta prueba.")
