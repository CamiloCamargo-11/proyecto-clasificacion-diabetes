# Proyecto de clasificación académica de diabetes

Aplicación Streamlit construida para acompañar el notebook final del proyecto de Minería de Datos.

## Archivos necesarios

En la carpeta raíz del repositorio deben estar:

- `app.py`
- `requirements.txt`
- `pipeline_diabetes.joblib` (generado por la sección 10 del notebook)
- `metricas_modelo.json` (generado por la sección 10)
- `caso_prueba_conocido.json` (generado por la sección 10)
- `notebook final corte 3.ipynb` (el notebook final ejecutado)

## Preparación

1. Ejecuta todas las celdas del notebook en orden.
2. Confirma que la sección 11 indique `PRUEBA SUPERADA`.
3. Copia los tres archivos generados (`pipeline_diabetes.joblib`, `metricas_modelo.json` y `caso_prueba_conocido.json`) a esta carpeta.
4. Sube todos los archivos a un repositorio público de GitHub.
5. En Streamlit Community Cloud, selecciona el repositorio, la rama `main` y `app.py` como archivo principal.

## Importante

El archivo `.joblib` debe ser el producido por el notebook final. No entrenes otro modelo dentro de `app.py`. Las versiones de NumPy, pandas, scikit-learn y joblib se fijan de acuerdo con las versiones impresas en el notebook. Si Streamlit informa que una versión no está disponible o hay incompatibilidad, revisa los registros antes de cambiar versiones.

Esta aplicación tiene fines académicos y no es una herramienta de diagnóstico médico.
