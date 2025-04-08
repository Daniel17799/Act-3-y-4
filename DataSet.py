import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta


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
print(df_transporte.head())


df_transporte.to_csv('datos_transporte.csv', index=False, encoding='utf-8')

print("\nDataFrame guardado exitosamente en 'datos_transporte_simulado.csv'")