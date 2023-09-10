# Etapa 2 - Diseño de sistemas de aprendizaje automático
# Asignatura: Análisis de Datos
# Grupo: 202016908_46
# Septiembre 2023

#Ejercicio Regresión Logística
# Dataset taken from: https://www.kaggle.com/datasets/dileep070/heart-disease-prediction-using-logistic-regression


# Paso 1 - importar librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# paso 2 - cargar dataset
data = pd.read_csv('data_regresion_logistica.csv')
data.head()
data.isnull().sum()

# Procesamiento de datos
# preprocesamiento de datos "datos vacios/nulos, variables categoricas, etc.".
data.fillna(0, inplace=True)

# Dividir los datos en características (X) y variables objetivo (y)
X = data.drop(columns=["TenYearCHD"])
y = data["TenYearCHD"]

# dividir la data en sets de entrenamiento y pruebas
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de regresion logistica
model = LogisticRegression()
model.fit(X_train, y_train)

# hacer predicciones en el set de pruebas
y_pred = model.predict(X_test)

# evaluar el modelo - "metricas"
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Mostrar los resultados
print("Interseccion (b)", model.intercept_)
print("Pendiente (m)", model.coef_)
print("Accuracy:", accuracy)
print("Matriz de Confusion:\n", conf_matrix)
print("Reporte de clasificación:\n", classification_rep)

# Visualizar la matriz de confusion
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=["No Enfermedad cardíaca", "Enfermedad cardíaca"], yticklabels=["No Enfermedad cardíaca", "Enfermedad cardíaca"])
plt.xlabel("Predicción")
plt.ylabel("Actual")
plt.title("Matriz de Confusion")
plt.show()
