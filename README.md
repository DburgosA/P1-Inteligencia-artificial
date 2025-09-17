# 🔬 Clasificación de Píxeles para Segmentación de Lesiones Dermatológicas

## 📋 Descripción del Proyecto

Este proyecto implementa y compara diferentes técnicas de clasificación a nivel de píxel para la segmentación automática de lesiones en imágenes dermatoscópicas del dataset ISIC. Se evalúan tres enfoques principales:

- **Clasificadores Bayesianos con características RGB**
- **Clasificadores Bayesianos con reducción de dimensionalidad (PCA)**  
- **Clustering no supervisado (K-Means)**

## 📁 Estructura del Proyecto

```
P1-Inteligencia-artificial/
├── 🔬 main.py                    # Pipeline experimental completo
├── 🖥️ gui_dermatoscopy.py        # Interfaz gráfica para diagnóstico
├── 📊 informe.tex               # Reporte académico LaTeX
├── 📂 src/                      # Módulos del sistema
│   ├── bayesian_classifier.py  # Clasificador bayesiano
│   ├── preprocessing.py         # Preprocesamiento de datos
│   ├── metrics.py              # Cálculo de métricas
│   ├── roc_analysis.py         # Análisis ROC
│   └── experiment_runner.py    # Ejecutor de experimentos
├── 📂 dataset/                  # Imágenes dermatoscópicas
│   ├── ISIC_*.jpg              # Imágenes originales
│   └── ISIC_*_expert.png       # Máscaras de segmentación
├── 📂 results/                  # Resultados experimentales
│   ├── report_data.json        # Datos del experimento
│   ├── *.png                   # Gráficos y visualizaciones
│   └── *.csv                   # Tablas de resultados
└── 📂 xarchivos/               # Archivos auxiliares
    └── requerimientos.tex      # Especificaciones del proyecto
```

## 🚀 Archivos Principales y Ejecución

### 1. 🔬 **main.py** - Pipeline Experimental Completo

**Propósito:** Ejecuta todo el pipeline de investigación científica con evaluación rigurosa de los tres métodos de clasificación.

**Funcionalidades:**
- ✅ Carga y exploración del dataset (150 imágenes ISIC)
- ✅ Partición estratificada (60% entrenamiento, 20% validación, 20% test)
- ✅ Preprocesamiento con normalización y PCA
- ✅ Entrenamiento de clasificadores bayesianos (RGB y PCA)
- ✅ Clustering K-Means no supervisado
- ✅ Análisis ROC y calibración de umbrales
- ✅ Evaluación completa con métricas a nivel píxel e imagen
- ✅ Generación de visualizaciones y reportes

**Cómo ejecutar:**
```bash
# Ejecutar experimento completo (duración: ~3 minutos)
python main.py
```

**Resultados generados:**
- `results/report_data.json` - Datos completos del experimento
- `results/*.png` - Gráficos ROC, comparaciones, ejemplos
- `results/*.csv` - Tablas de métricas finales

---

### 2. 🖥️ **gui_dermatoscopy.py** - Interfaz Gráfica de Diagnóstico

**Propósito:** Aplicación con interfaz gráfica para análisis interactivo de imágenes dermatoscópicas en tiempo real.

**Funcionalidades:**
- 🖼️ Carga de imágenes dermatoscópicas individuales
- 🎛️ Selección de clasificador (Bayesiano RGB/PCA)
- ⚙️ Configuración de criterios de umbral (Youden, EER, Alta Sensibilidad)
- 📊 Análisis en tiempo real con visualización
- 💾 Guardado de resultados y reportes
- 🧠 Re-entrenamiento incremental con nuevas imágenes
- 🚀 Auto-entrenamiento con 90 imágenes del dataset

**Cómo ejecutar:**
```bash
# Abrir interfaz gráfica
python gui_dermatoscopy.py
```

**Instrucciones de uso:**
1. **Cargar Imagen:** Seleccionar imagen dermatoscópica (.jpg, .png)
2. **Configurar:** Elegir clasificador y criterio de umbral
3. **Analizar:** Hacer clic en "Analizar Imagen"
4. **Revisar:** Examinar máscara predicha y estadísticas
5. **Guardar:** Exportar resultados si es necesario

**Funciones Avanzadas:**
- **Cargar Máscara Real:** Para corrección manual y re-entrenamiento
- **Re-entrenar Modelo:** Mejora incremental con feedback del usuario
- **Auto-Entrenar:** Entrenamiento automático con todo el dataset

---

## 📚 Módulos del Sistema (src/)

### 🧠 **bayesian_classifier.py**
Implementa clasificadores bayesianos gaussianos con:
- Estimación de parámetros por máxima verosimilitud
- Cálculo de probabilidades posteriores
- Calibración de umbrales de decisión

### 🔧 **preprocessing.py**
Herramientas de preprocesamiento:
- Normalización por canal RGB
- Análisis de componentes principales (PCA)
- Muestreo equilibrado de clases

### 📊 **metrics.py**
Cálculo de métricas de evaluación:
- Métricas a nivel píxel (exactitud, precisión, sensibilidad, especificidad)
- Métricas a nivel imagen (índice de Jaccard)
- Matrices de confusión

### 📈 **roc_analysis.py**
Análisis de curvas ROC:
- Cálculo de AUC
- Selección de umbrales óptimos (Youden, EER)
- Generación de curvas ROC comparativas

### ⚙️ **experiment_runner.py**
Coordinador de experimentos:
- Gestión del pipeline completo
- Paralelización de tareas
- Generación de reportes

## 🔧 Requisitos del Sistema

### Dependencias Python:
```bash
pip install numpy pandas scikit-learn matplotlib opencv-python pillow tkinter
```

### Requisitos mínimos:
- **Python:** 3.8+
- **RAM:** 4GB (recomendado 8GB)
- **Almacenamiento:** 500MB para dataset y resultados
- **SO:** Windows 10/11, macOS, Linux

## 📊 Resultados Esperados

### Métricas de Desempeño:
- **K-Means:** 92.61% exactitud, F1-Score 0.8586
- **Bayesiano RGB:** 84.50% exactitud, AUC 0.8896
- **Bayesiano PCA:** 84.56% exactitud, AUC 0.8798

### Tiempo de Ejecución:
- **main.py:** ~3 minutos (experimento completo)
- **gui_dermatoscopy.py:** <5 segundos por imagen

## 🎯 Casos de Uso

### 🔬 **Para Investigación (main.py):**
- Evaluación rigurosa de métodos
- Generación de métricas comparativas
- Análisis estadístico robusto
- Reproducibilidad científica

### 🏥 **Para Aplicación Clínica (gui_dermatoscopy.py):**
- Análisis interactivo de lesiones
- Diagnóstico asistido por IA
- Entrenamiento adaptativo
- Interface amigable para médicos

## 🚨 Solución de Problemas

### Error: "No se encontró el dataset"
```bash
# Verificar estructura de directorios
ls dataset/
# Debe contener archivos ISIC_*.jpg e ISIC_*_expert.png
```

### Error: "Módulos no encontrados"
```bash
# Instalar dependencias
pip install -r requirements.txt
# O instalar manualmente:
pip install numpy pandas scikit-learn matplotlib opencv-python pillow
```

### Error: "Clasificadores no entrenados"
```bash
# Ejecutar primero el experimento completo
python main.py
# Luego ejecutar la interfaz gráfica
python gui_dermatoscopy.py
```

## 📄 Documentación Adicional

- **📋 Informe Académico:** `informe.tex` (LaTeX)
- **📋 Requerimientos:** `xarchivos/requerimientos.tex`
- **📊 Resultados:** `results/` (generados automáticamente)

## 👥 Contribuciones

Este proyecto fue desarrollado como parte del curso de Inteligencia Artificial, implementando técnicas de aprendizaje automático para segmentación médica con enfoque en dermatología computacional.

---

## 🏃‍♂️ Inicio Rápido

```bash
# 1. Clonar repositorio
git clone [URL_del_repositorio]
cd P1-Inteligencia-artificial

# 2. Instalar dependencias
pip install numpy pandas scikit-learn matplotlib opencv-python pillow

# 3. Ejecutar experimento completo
python main.py

# 4. Abrir interfaz gráfica
python gui_dermatoscopy.py
```

**¡Listo para analizar lesiones dermatológicas con IA! 🎉**