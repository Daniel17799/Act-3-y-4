#Importamos nuestras depedencias:
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

#Ejecutamos la simulacón de nuestro DataSet

np.random.seed(42)
random.seed(42)


estaciones = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
num_viajes = 500

data = []
for _ in range(num_viajes):
    origen = random.choice(estaciones)
    destino = random.choice(estaciones)
    if origen == destino:
        continue

    hora_inicio = datetime(2025, 4, 7, random.randint(6, 22), random.randint(0, 59), random.randint(0, 59))


    distancia = abs(estaciones.index(destino) - estaciones.index(origen))
    tiempo_base = distancia * 5 + random.uniform(-2, 5)


    if 7 <= hora_inicio.hour < 9 or 17 <= hora_inicio.hour < 19:
        tiempo_base *= random.uniform(1.2, 1.8)

   
    if hora_inicio.weekday() == 4: # Viernes
        tiempo_base *= random.uniform(1.1, 1.4)

  
    if random.random() < 0.2: 
        tiempo_base *= random.uniform(1.15, 1.3)

    tiempo_viaje = max(5, int(tiempo_base)) 

    data.append({
        'origen': origen,
        'destino': destino,
        'hora_inicio': hora_inicio,
        'dia_semana': hora_inicio.weekday(), # 0: Lunes, 6: Domingo
        'es_hora_pico': 1 if (7 <= hora_inicio.hour < 9 or 17 <= hora_inicio.hour < 19) else 0,
        'distancia_estaciones': distancia,
        'lluvia': 1 if random.random() < 0.2 else 0,
        'tiempo_viaje_real': tiempo_viaje
    })

df_transporte = pd.DataFrame(data)



# Codificar variables categóricas (origen y destino)
label_encoder_origen = LabelEncoder()
df_transporte['origen_encoded'] = label_encoder_origen.fit_transform(df_transporte['origen'])

label_encoder_destino = LabelEncoder()
df_transporte['destino_encoded'] = label_encoder_destino.fit_transform(df_transporte['destino'])

# Seleccionar las características (variables predictoras)
features = ['origen_encoded', 'destino_encoded', 'dia_semana', 'es_hora_pico', 'distancia_estaciones', 'lluvia', 'hora_inicio']

# Extraer características relevantes de la columna 'hora_inicio'
df_transporte['hora'] = df_transporte['hora_inicio'].dt.hour
df_transporte['minuto'] = df_transporte['hora_inicio'].dt.minute
features.remove('hora_inicio')
features.extend(['hora', 'minuto'])

# Variable objetivo
target = 'tiempo_viaje_real'

# Dividir los datos en conjuntos de entrenamiento y prueba
X = df_transporte[features]
y = df_transporte[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Conjunto de entrenamiento:", X_train.shape, y_train.shape)
print("Conjunto de prueba:", X_test.shape, y_test.shape)

# 2. Selección y Entrenamiento del Modelo

# Elegimos un modelo de Regresión de Bosques Aleatorios 
modelo = RandomForestRegressor(n_estimators=100, random_state=42) # n_estimators: número de árboles en el bosque
modelo.fit(X_train, y_train)

# 3. Evaluación del Modelo

# Se realiza predicciones en el conjunto de prueba
predicciones = modelo.predict(X_test)

# Evaluar el rendimiento del modelo utilizando métricas relevantes para la regresión
mse = mean_squared_error(y_test, predicciones)
r2 = r2_score(y_test, predicciones)

print(f"\nError Cuadratico Medio (MSE): {mse:.2f}")
print(f"Coeficiente de Determinacion (R^2): {r2:.2f}")



# Ejemplo 
nuevo_viaje = pd.DataFrame({
    'origen': ['B'],
    'destino': ['G'],
    'hora_inicio': [datetime(2025, 4, 7, 18, 15, 0)],
    'dia_semana': [0], # Lunes
    'es_hora_pico': [1],
    'distancia_estaciones': [abs(estaciones.index('G') - estaciones.index('B'))],
    'lluvia': [0]
})

# Preprocesar los datos del nuevo viaje de la misma manera que los datos de entrenamiento
nuevo_viaje['origen_encoded'] = label_encoder_origen.transform(nuevo_viaje['origen'])
nuevo_viaje['destino_encoded'] = label_encoder_destino.transform(nuevo_viaje['destino'])
nuevo_viaje['hora'] = nuevo_viaje['hora_inicio'].dt.hour
nuevo_viaje['minuto'] = nuevo_viaje['hora_inicio'].dt.minute
nuevo_viaje_features = nuevo_viaje[['origen_encoded', 'destino_encoded', 'dia_semana', 'es_hora_pico', 'distancia_estaciones', 'lluvia', 'hora', 'minuto']]

prediccion_nuevo_viaje = modelo.predict(nuevo_viaje_features)
print(f"\nPredicción del tiempo de viaje para el nuevo escenario: {prediccion_nuevo_viaje[0]:.2f} minutos")