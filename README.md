# Predicción de Deserción de Clientes – Model Fitness

##  Descripción del Proyecto
Este proyecto predice la deserción de clientes en una cadena de gimnasios utilizando técnicas de Machine Learning supervisado y no supervisado. El objetivo es identificar qué clientes tienen mayor probabilidad de abandonar el servicio, permitiendo implementar estrategias de retención antes de perderlos.

---
##  Objetivos
- Predecir la deserción de clientes mediante modelos de clasificación
- Segmentar clientes por perfil de comportamiento usando clustering
- Identificar las variables clave que impulsan el abandono
- Generar recomendaciones accionables para reducir la tasa de deserción

---
## 🛠️ Herramientas y Librerías

| Librería | Uso |
|---|---|
| `Pandas` | Manipulación y análisis de datos |
| `NumPy` | Operaciones numéricas |
| `Scikit-learn` | Modelos de Machine Learning y preprocesamiento |
| `Matplotlib / Seaborn` | Visualización de datos |
| `SciPy` | Clustering jerárquico (whiten, pdist, linkage) |

---
##  Metodología

### 1. Análisis Exploratorio de Datos (EDA)
- Identificación de valores nulos, duplicados y tipos de datos
- Análisis de valores medios agrupados por estado de deserción
- Matriz de correlación para detectar multicolinealidad
- Visualización de distribución de clientes activos vs desertores por mes de contrato

### 2. Preprocesamiento de Datos
- Separación de variables **numéricas** y **binarias**
- Aplicación de **StandardScaler** para normalizar variables numéricas (media = 0, desviación estándar = 1)
- Concatenación de tablas escaladas para construir el dataset final listo para modelar

### 3. Aprendizaje Supervisado – Modelos de Clasificación

#### 🔹 Regresión Logística
Modelo estadístico que estima la probabilidad de un resultado binario (deserción o no deserción) a partir de las variables de entrada. Encuentra la frontera lineal óptima para separar ambas clases.

**Resultados:**
- Exactitud (Accuracy): **93%**
- Precisión: **94%**
- Recall: **97%**
- Verdaderos Positivos (clientes retenidos correctamente identificados): **580**

####  Random Forest
Método de ensamble que construye múltiples árboles de decisión sobre subconjuntos aleatorios de datos y combina sus predicciones. Reduce el sobreajuste y mejora la generalización del modelo.

**Resultados:**
- Exactitud (Accuracy): **90%**
- Precisión: **93%**
- Recall: **94%**
- Verdaderos Positivos (clientes retenidos correctamente identificados): **574**

### 4. Aprendizaje No Supervisado – Segmentación de Clientes

####  Clustering Jerárquico (Dendrograma)
Se utilizaron los métodos Ward y Average con distancia Euclidiana para explorar la estructura natural de agrupamiento antes de aplicar K-Means.

####  K-Means
Se agruparon los clientes en **5 segmentos de comportamiento**:

| Clúster | Perfil |
|---|---|
| Clúster 0 | Estable – bajo riesgo de deserción |
| Clúster 1 | ⚠️ Alto riesgo de deserción – principal grupo de abandono |
| Clúster 2 | Riesgo moderado de deserción |
| Clúster 3 | Muy fiel – casi cero deserción |
| Clúster 4 | Estable – alta retención |

---

## 📊 Hallazgos Principales
- El **período de contrato** y la **frecuencia de clases** son los predictores más fuertes de deserción
- Clientes con **contratos cortos (1 mes)** y **baja frecuencia de clases** tienen mayor probabilidad de abandono
- El **Clúster 1** concentra el mayor riesgo de deserción y debe ser el principal objetivo de campañas de retención
- La **Regresión Logística** superó al Random Forest con un 93% de exactitud

---

## ✅ Recomendaciones
- Ofrecer incentivos para convertir contratos mensuales en planes de mayor duración
- Diseñar campañas de reactivación dirigidas a clientes del Clúster 1
- Monitorear la frecuencia de clases como señal temprana de riesgo de deserción
- Priorizar esfuerzos de retención en los primeros 3 meses de vida del cliente

