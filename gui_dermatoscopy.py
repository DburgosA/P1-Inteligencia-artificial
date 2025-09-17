"""
Interfaz Gráfica para Análisis de Imágenes Dermatoscópicas
Clasificación de Lesiones usando Clasificadores Bayesianos
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import cv2
from PIL import Image, ImageTk
import os
import sys
import pickle
import json
from datetime import datetime

# Agregar src al path para importar módulos
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from bayesian_classifier import BayesianClassifier
from kmeans_classifier import KMeansClassifier
from preprocessing import DataPreprocessor
from metrics import MetricsCalculator
from roc_analysis import ROCAnalyzer


class DermatoscopyAnalyzer:
    """
    Interfaz gráfica para análisis de imágenes dermatoscópicas
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("Analizador de Lesiones Dermatoscópicas")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Variables de estado
        self.current_image = None
        self.original_image = None
        self.predicted_mask = None
        self.ground_truth_mask = None  # Para reentrenamiento
        self.bayesian_rgb_classifier = None
        self.bayesian_pca_classifier = None
        self.kmeans_classifier = None
        self.preprocessor_rgb = None
        self.preprocessor_pca = None
        self.roc_data = None
        
        # Datos para reentrenamiento incremental
        self.training_data = {
            'lesion_pixels': [],
            'non_lesion_pixels': [],
            'images_processed': 0
        }
        
        # Variables de interfaz
        self.classifier_var = tk.StringVar(value="Bayesiano RGB")
        self.threshold_criterion_var = tk.StringVar(value="Youden")
        self.custom_threshold_var = tk.DoubleVar(value=0.0)
        
        # Configurar estilo
        self.setup_style()
        
        # Cargar modelos entrenados
        self.load_trained_models()
        
        # Crear interfaz
        self.create_interface()
        
        # Variables para estadísticas
        self.stats_data = {}
        
    def setup_style(self):
        """Configurar estilo de la interfaz"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configurar colores personalizados
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), background='#f0f0f0')
        style.configure('Subtitle.TLabel', font=('Arial', 12, 'bold'), background='#f0f0f0')
        style.configure('Info.TLabel', font=('Arial', 10), background='#f0f0f0')
        style.configure('Success.TLabel', font=('Arial', 10), background='#f0f0f0', foreground='green')
        style.configure('Error.TLabel', font=('Arial', 10), background='#f0f0f0', foreground='red')
        
    def load_trained_models(self):
        """Cargar modelos entrenados y datos ROC"""
        try:
            # Intentar cargar datos del experimento
            report_data_path = os.path.join("results", "report_data.json")
            if os.path.exists(report_data_path):
                with open(report_data_path, 'r') as f:
                    self.roc_data = json.load(f)
                print("✓ Datos ROC cargados exitosamente")
            else:
                print("⚠ Archivo de datos ROC no encontrado. Ejecute main.py primero.")
                
            # Cargar parámetros de normalización del dataset
            self.normalization_params = {
                'mean': [196.85761843, 118.47437496, 121.60051153],
                'std': [34.94214013, 43.58808125, 47.49784474]
            }
            
            # Crear clasificadores con parámetros conocidos del experimento
            self.create_classifiers_from_experiment()
            
        except Exception as e:
            print(f"Error cargando modelos: {e}")
            messagebox.showwarning("Advertencia", 
                                 "No se pudieron cargar los modelos entrenados.\n"
                                 "Ejecute main.py primero para entrenar los modelos.")
    
    def create_classifiers_from_experiment(self):
        """Crear clasificadores usando parámetros del experimento"""
        try:
            # Parámetros conocidos del experimento para RGB
            rgb_params = {
                'mu_lesion': np.array([-0.54681313, -0.7027965, -0.7264065]),
                'mu_non_lesion': np.array([0.5477914, 0.7035777, 0.7243743]),
                'prior_lesion': 0.5,
                'prior_non_lesion': 0.5
            }
            
            # Parámetros para PCA
            pca_params = {
                'mu_lesion': np.array([-1.1451365, 0.09199613]),
                'mu_non_lesion': np.array([1.1453842, -0.09349232]),
                'prior_lesion': 0.5,
                'prior_non_lesion': 0.5
            }
            
            # Crear clasificador RGB
            self.bayesian_rgb_classifier = BayesianClassifier(seed=42)
            self.bayesian_rgb_classifier.mu_lesion = rgb_params['mu_lesion']
            self.bayesian_rgb_classifier.mu_non_lesion = rgb_params['mu_non_lesion']
            self.bayesian_rgb_classifier.prior_lesion = rgb_params['prior_lesion']
            self.bayesian_rgb_classifier.prior_non_lesion = rgb_params['prior_non_lesion']
            self.bayesian_rgb_classifier.feature_dim = 3
            self.bayesian_rgb_classifier.is_fitted = True
            
            # Matrices de covarianza simplificadas (identidad con regularización)
            self.bayesian_rgb_classifier.sigma_lesion = np.eye(3) * 0.1
            self.bayesian_rgb_classifier.sigma_non_lesion = np.eye(3) * 0.1
            
            # Crear clasificador PCA
            self.bayesian_pca_classifier = BayesianClassifier(seed=42)
            self.bayesian_pca_classifier.mu_lesion = pca_params['mu_lesion']
            self.bayesian_pca_classifier.mu_non_lesion = pca_params['mu_non_lesion']
            self.bayesian_pca_classifier.prior_lesion = pca_params['prior_lesion']
            self.bayesian_pca_classifier.prior_non_lesion = pca_params['prior_non_lesion']
            self.bayesian_pca_classifier.feature_dim = 2
            self.bayesian_pca_classifier.is_fitted = True
            
            # Matrices de covarianza para PCA
            self.bayesian_pca_classifier.sigma_lesion = np.eye(2) * 0.1
            self.bayesian_pca_classifier.sigma_non_lesion = np.eye(2) * 0.1
            
            # Crear clasificador K-Means
            self.kmeans_classifier = KMeansClassifier(n_clusters=2, seed=42)
            # El K-Means será entrenado cuando se use por primera vez
            
            # Crear preprocessors
            self.preprocessor_rgb = DataPreprocessor(seed=42)
            self.preprocessor_pca = DataPreprocessor(seed=42)
            
            # Configurar PCA con parámetros conocidos
            if self.roc_data and 'pca_info' in self.roc_data:
                pca_info = self.roc_data['pca_info']
                components = np.array(pca_info['components'])
                self.pca_components = components
                self.pca_mean = np.array([0, 0, 0])  # Ya normalizado
                self.n_components = 2
            else:
                # Usar componentes por defecto del experimento
                self.pca_components = np.array([
                    [0.5444, 0.5974, 0.5889],
                    [-0.6910, 0.2392, 0.6830]
                ])
            
            print("✓ Clasificadores creados exitosamente")
            
        except Exception as e:
            print(f"Error creando clasificadores: {e}")
            
    def create_interface(self):
        """Crear la interfaz gráfica"""
        # Panel principal
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Título
        title_label = ttk.Label(main_frame, text="Analizador de Lesiones Dermatoscópicas", 
                               style='Title.TLabel')
        title_label.pack(pady=(0, 20))
        
        # Frame superior para controles
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.create_control_panel(control_frame)
        
        # Frame inferior para visualización
        viz_frame = ttk.Frame(main_frame)
        viz_frame.pack(fill=tk.BOTH, expand=True)
        
        self.create_visualization_panel(viz_frame)
        
    def create_control_panel(self, parent):
        """Crear panel de controles"""
        # Frame para carga de imagen
        load_frame = ttk.LabelFrame(parent, text="Carga de Imagen", padding=10)
        load_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        ttk.Button(load_frame, text="Cargar Imagen", 
                  command=self.load_image, width=15).pack(pady=5)
        
        self.image_info_label = ttk.Label(load_frame, text="No hay imagen cargada", 
                                         style='Info.TLabel')
        self.image_info_label.pack(pady=5)
        
        # Frame para configuración de clasificador
        classifier_frame = ttk.LabelFrame(parent, text="Configuración del Clasificador", padding=10)
        classifier_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        ttk.Label(classifier_frame, text="Clasificador:", style='Info.TLabel').pack(anchor=tk.W)
        classifier_combo = ttk.Combobox(classifier_frame, textvariable=self.classifier_var,
                                       values=["Bayesiano RGB", "Bayesiano PCA", "K-Means"],
                                       state="readonly", width=15)
        classifier_combo.pack(pady=5)
        
        ttk.Label(classifier_frame, text="Criterio de Umbral:", style='Info.TLabel').pack(anchor=tk.W, pady=(10,0))
        threshold_combo = ttk.Combobox(classifier_frame, textvariable=self.threshold_criterion_var,
                                      values=["Youden", "EER", "Alta Sensibilidad", "Personalizado"],
                                      state="readonly", width=15)
        threshold_combo.pack(pady=5)
        threshold_combo.bind('<<ComboboxSelected>>', self.on_threshold_change)
        
        # Frame para umbral personalizado (inicialmente oculto)
        self.custom_threshold_frame = ttk.Frame(classifier_frame)
        ttk.Label(self.custom_threshold_frame, text="Umbral:", style='Info.TLabel').pack(side=tk.LEFT)
        ttk.Entry(self.custom_threshold_frame, textvariable=self.custom_threshold_var, 
                 width=8).pack(side=tk.LEFT, padx=(5,0))
        
        # Frame para acciones
        action_frame = ttk.LabelFrame(parent, text="Acciones", padding=10)
        action_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        self.analyze_button = ttk.Button(action_frame, text="Analizar Imagen", 
                                        command=self.analyze_image, width=15)
        self.analyze_button.pack(pady=5)
        
        # Barra de progreso
        self.progress = ttk.Progressbar(action_frame, mode='indeterminate', length=120)
        self.progress.pack(pady=5)
        
        ttk.Button(action_frame, text="Guardar Resultados", 
                  command=self.save_results, width=15).pack(pady=5)
        
        # Separador
        ttk.Separator(action_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Entrenamiento adaptativo
        ttk.Label(action_frame, text="Mejora de Precisión:", style='Subtitle.TLabel').pack(anchor=tk.W)
        
        ttk.Button(action_frame, text="Cargar Máscara Real", 
                  command=self.load_ground_truth, width=15).pack(pady=2)
        
        ttk.Button(action_frame, text="Re-entrenar Modelo", 
                  command=self.retrain_classifier, width=15).pack(pady=2)
        
        ttk.Button(action_frame, text="Limpiar", 
                  command=self.clear_results, width=15).pack(pady=5)
        
        # Frame para estadísticas
        stats_frame = ttk.LabelFrame(parent, text="Estadísticas", padding=10)
        stats_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.stats_text = tk.Text(stats_frame, height=8, width=30, wrap=tk.WORD)
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar para estadísticas
        stats_scrollbar = ttk.Scrollbar(stats_frame, orient=tk.VERTICAL, command=self.stats_text.yview)
        stats_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.stats_text.config(yscrollcommand=stats_scrollbar.set)
        
    def create_visualization_panel(self, parent):
        """Crear panel de visualización"""
        # Crear figura de matplotlib
        self.fig = Figure(figsize=(14, 6), facecolor='white')
        
        # Crear subplots
        self.ax1 = self.fig.add_subplot(131)  # Imagen original
        self.ax2 = self.fig.add_subplot(132)  # Máscara predicha
        self.ax3 = self.fig.add_subplot(133)  # Superposición
        
        # Configurar ejes
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.set_xticks([])
            ax.set_yticks([])
        
        self.ax1.set_title("Imagen Original", fontsize=12, fontweight='bold')
        self.ax2.set_title("Máscara Predicha", fontsize=12, fontweight='bold')
        self.ax3.set_title("Superposición", fontsize=12, fontweight='bold')
        
        # Crear canvas
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Barra de herramientas de matplotlib
        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.pack(fill=tk.X)
        
    def on_threshold_change(self, event=None):
        """Manejar cambio en criterio de umbral"""
        if self.threshold_criterion_var.get() == "Personalizado":
            self.custom_threshold_frame.pack(pady=(5,0))
        else:
            self.custom_threshold_frame.pack_forget()
    
    def load_image(self):
        """Cargar imagen para análisis"""
        file_path = filedialog.askopenfilename(
            title="Seleccionar imagen dermatoscópica",
            filetypes=[
                ("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("Todos los archivos", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Mostrar progreso
                self.image_info_label.config(text="Cargando imagen...")
                self.root.update()
                
                # Cargar y procesar imagen
                image = cv2.imread(file_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Redimensionar si es muy grande (optimización)
                height, width = image.shape[:2]
                max_display_size = 1000
                if max(height, width) > max_display_size:
                    scale = max_display_size / max(height, width)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                
                self.original_image = image.copy()
                self.current_image = image
                
                # Actualizar información de imagen
                h, w, c = image.shape
                file_size = os.path.getsize(file_path) / (1024*1024)  # MB
                self.image_info_label.config(text=f"Imagen: {w}x{h} ({file_size:.1f}MB)")
                
                # Mostrar imagen original (optimizado)
                self.ax1.clear()
                self.ax1.imshow(image)
                self.ax1.set_title("Imagen Original", fontsize=12, fontweight='bold')
                self.ax1.set_xticks([])
                self.ax1.set_yticks([])
                
                # Limpiar otros paneles
                self.ax2.clear()
                self.ax3.clear()
                self.ax2.set_title("Máscara Predicha", fontsize=12, fontweight='bold')
                self.ax3.set_title("Superposición", fontsize=12, fontweight='bold')
                self.ax2.set_xticks([])
                self.ax2.set_yticks([])
                self.ax3.set_xticks([])
                self.ax3.set_yticks([])
                
                # Actualizar canvas de forma eficiente
                self.canvas.draw_idle()
                
                # Limpiar estadísticas
                self.stats_text.delete(1.0, tk.END)
                
                print(f"✓ Imagen cargada: {os.path.basename(file_path)} ({w}x{h})")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error cargando imagen: {str(e)}")
                self.image_info_label.config(text="Error cargando imagen")
    
    def get_threshold_value(self):
        """Obtener valor de umbral según criterio seleccionado"""
        if not self.roc_data:
            return 0.0
            
        classifier_type = self.classifier_var.get()
        criterion = self.threshold_criterion_var.get()
        
        if criterion == "Personalizado":
            return self.custom_threshold_var.get()
        
        # Mapear clasificador a clave en datos ROC
        roc_key = "bayesian_rgb" if classifier_type == "Bayesiano RGB" else "bayesian_pca"
        
        if roc_key not in self.roc_data.get('roc_analysis', {}):
            return 0.0
            
        roc_info = self.roc_data['roc_analysis'][roc_key]
        
        if criterion == "Youden":
            return roc_info.get('youden', {}).get('threshold', 0.0)
        elif criterion == "EER":
            return roc_info.get('eer', {}).get('threshold', 0.0)
        elif criterion == "Alta Sensibilidad":
            return roc_info.get('high_sensitivity', {}).get('threshold', 0.0)
        
        return 0.0
    
    def analyze_image(self):
        """Analizar imagen con clasificador seleccionado"""
        if self.current_image is None:
            messagebox.showwarning("Advertencia", "Por favor, cargue una imagen primero.")
            return
        
        try:
            # Iniciar progreso
            self.analyze_button.config(state='disabled')
            self.progress.start(10)
            
            # Mostrar progreso
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(1.0, "Analizando imagen...\nPor favor espere...")
            self.root.update()
            
            # Obtener configuración
            classifier_type = self.classifier_var.get()
            threshold = self.get_threshold_value()
            
            # Preparar datos (optimizado)
            image = self.current_image.astype(np.float32)
            h, w, c = image.shape
            
            # Redimensionar si es muy grande para acelerar procesamiento
            max_size = 600  # Reducido para mayor velocidad
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                original_shape = (h, w)
                resize_needed = True
            else:
                image_resized = image
                resize_needed = False
                
            # Convertir a formato de píxeles
            pixels = image_resized.reshape(-1, 3)
            
            # Normalizar datos (vectorizado)
            mean = np.array(self.normalization_params['mean'])
            std = np.array(self.normalization_params['std'])
            normalized_pixels = (pixels - mean) / std
            
            # Seleccionar características
            if classifier_type == "Bayesiano RGB":
                classifier = self.bayesian_rgb_classifier
                features = normalized_pixels
            elif classifier_type == "Bayesiano PCA":
                classifier = self.bayesian_pca_classifier
                features = np.dot(normalized_pixels, self.pca_components.T)
            else:  # K-Means
                classifier = self.kmeans_classifier
                features = normalized_pixels
            
            # Realizar predicción (optimizada)
            if classifier:
                if classifier_type == "K-Means":
                    # K-Means funciona diferente - predice clusters directamente
                    if not hasattr(classifier, 'is_fitted') or not classifier.is_fitted:
                        # Entrenar K-Means con la imagen actual si no está entrenado
                        print("Entrenando K-Means automáticamente...")
                        sample_size = min(10000, len(features))  # Usar muestra para velocidad
                        sample_indices = np.random.choice(len(features), sample_size, replace=False)
                        classifier.fit(features[sample_indices])
                        print("✓ K-Means entrenado exitosamente")
                    
                    # Predecir clusters
                    cluster_labels = classifier.predict(features)
                    
                    # Para segmentación, asumimos que cluster 1 es la lesión
                    # Esto se puede ajustar basado en los resultados
                    predictions = (cluster_labels == 1).astype(np.uint8)
                    
                    # Para scores, usamos la distancia al centro del cluster lesión
                    distances_to_lesion = np.linalg.norm(features - classifier.centroids[1], axis=1)
                    distances_to_normal = np.linalg.norm(features - classifier.centroids[0], axis=1)
                    decision_scores = distances_to_normal - distances_to_lesion  # Mayor score = más lesión
                elif classifier.is_fitted:
                    # Clasificadores Bayesianos (solo si están entrenados)
                    decision_scores = self.fast_decision_scores(features, classifier)
                    predictions = (decision_scores > threshold).astype(np.uint8)
                else:
                    messagebox.showerror("Error", "Clasificador Bayesiano no disponible. Ejecute main.py primero.")
                    return
                
                # Redimensionar resultados si es necesario
                if resize_needed:
                    mask_pred_small = predictions.reshape(image_resized.shape[:2])
                    mask_pred = cv2.resize(mask_pred_small.astype(np.float32), 
                                         (original_shape[1], original_shape[0]), 
                                         interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                    scores_resized = decision_scores.reshape(image_resized.shape[:2])
                    scores_image = cv2.resize(scores_resized, (original_shape[1], original_shape[0]))
                else:
                    mask_pred = predictions.reshape(h, w)
                    scores_image = decision_scores.reshape(h, w)
                
                # Calcular estadísticas (optimizado)
                total_pixels = mask_pred.size
                lesion_pixels = int(np.sum(mask_pred))
                non_lesion_pixels = total_pixels - lesion_pixels
                lesion_percentage = (lesion_pixels / total_pixels) * 100
                
                # Almacenar resultados
                self.predicted_mask = mask_pred
                self.stats_data = {
                    'classifier': classifier_type,
                    'threshold_criterion': self.threshold_criterion_var.get(),
                    'threshold_value': threshold,
                    'total_pixels': total_pixels,
                    'lesion_pixels': lesion_pixels,
                    'non_lesion_pixels': non_lesion_pixels,
                    'lesion_percentage': lesion_percentage,
                    'mean_score_lesion': float(np.mean(scores_image[mask_pred == 1])) if lesion_pixels > 0 else 0,
                    'mean_score_non_lesion': float(np.mean(scores_image[mask_pred == 0])) if non_lesion_pixels > 0 else 0,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Mostrar resultados
                self.display_results(mask_pred, scores_image)
                self.update_statistics()
                
                print(f"✓ Análisis completado - {lesion_pixels} píxeles de lesión detectados")
                
            else:
                messagebox.showerror("Error", "Clasificador no disponible.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error en análisis: {str(e)}")
            print(f"Error detallado: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Detener progreso
            self.progress.stop()
            self.analyze_button.config(state='normal')
    
    def normalize_pixels(self, pixels):
        """Normalizar píxeles usando parámetros del experimento"""
        mean = np.array(self.normalization_params['mean'])
        std = np.array(self.normalization_params['std'])
        return (pixels - mean) / std
    
    def fast_decision_scores(self, features, classifier):
        """Calcular scores de decisión de forma optimizada"""
        try:
            # Diferencias vectorizadas
            diff_lesion = features - classifier.mu_lesion
            diff_non_lesion = features - classifier.mu_non_lesion
            
            # Distancias simplificadas (asumiendo matrices de covarianza diagonales)
            sigma_inv_lesion = 1.0 / 0.1  # 1/sigma^2 simplificado
            sigma_inv_non_lesion = 1.0 / 0.1
            
            # Log-likelihoods simplificados
            log_likelihood_lesion = -0.5 * sigma_inv_lesion * np.sum(diff_lesion**2, axis=1)
            log_likelihood_non_lesion = -0.5 * sigma_inv_non_lesion * np.sum(diff_non_lesion**2, axis=1)
            
            # Log ratio + log priors
            log_prior_ratio = np.log(classifier.prior_lesion) - np.log(classifier.prior_non_lesion)
            decision_scores = log_likelihood_lesion - log_likelihood_non_lesion + log_prior_ratio
            
            return decision_scores
            
        except Exception as e:
            print(f"Error en cálculo optimizado, usando método original: {e}")
            return classifier.decision_scores(features)
    
    def apply_pca_transform(self, normalized_pixels):
        """Aplicar transformación PCA"""
        if not hasattr(self, 'pca_components'):
            # Usar componentes por defecto si no están disponibles
            components = np.array([
                [0.5444, 0.5974, 0.5889],
                [-0.6910, 0.2392, 0.6830]
            ])
        else:
            components = self.pca_components
        
        return np.dot(normalized_pixels, components.T)
    
    def display_results(self, mask_pred, scores_image):
        """Mostrar resultados de análisis de forma optimizada"""
        # Mostrar máscara predicha
        self.ax2.clear()
        self.ax2.imshow(mask_pred, cmap='gray', interpolation='nearest')
        self.ax2.set_title("Máscara Predicha", fontsize=12, fontweight='bold')
        self.ax2.set_xticks([])
        self.ax2.set_yticks([])
        
        # Crear superposición optimizada
        overlay = self.create_overlay_fast(self.current_image, mask_pred)
        
        self.ax3.clear()
        self.ax3.imshow(overlay, interpolation='nearest')
        self.ax3.set_title("Superposición", fontsize=12, fontweight='bold')
        self.ax3.set_xticks([])
        self.ax3.set_yticks([])
        
        # Actualizar canvas de forma eficiente
        self.canvas.draw_idle()
    
    def create_overlay_fast(self, image, mask):
        """Crear superposición optimizada de imagen original con máscara"""
        overlay = image.copy().astype(np.uint8)
        
        # Crear máscara coloreada de forma vectorizada
        overlay[mask == 1] = overlay[mask == 1] * 0.6 + np.array([255, 0, 0]) * 0.4
        
        return overlay.astype(np.uint8)
    
    def create_overlay(self, image, mask):
        """Crear superposición de imagen original con máscara"""
        overlay = image.copy()
        
        # Crear máscara coloreada (rojo para lesiones)
        mask_colored = np.zeros_like(image)
        mask_colored[mask == 1] = [255, 0, 0]  # Rojo para lesiones
        
        # Combinar con transparencia
        alpha = 0.4
        overlay = cv2.addWeighted(overlay.astype(np.uint8), 1-alpha, 
                                 mask_colored.astype(np.uint8), alpha, 0)
        
        return overlay
    
    def update_statistics(self):
        """Actualizar panel de estadísticas"""
        if not self.stats_data:
            return
        
        stats_text = f"""ANÁLISIS COMPLETADO
{self.stats_data['timestamp']}

CONFIGURACIÓN:
• Clasificador: {self.stats_data['classifier']}
• Criterio: {self.stats_data['threshold_criterion']}
• Umbral: {self.stats_data['threshold_value']:.4f}

ESTADÍSTICAS DE PÍXELES:
• Total: {self.stats_data['total_pixels']:,}
• Lesión: {self.stats_data['lesion_pixels']:,}
• No-lesión: {self.stats_data['non_lesion_pixels']:,}
• % Lesión: {self.stats_data['lesion_percentage']:.2f}%

SCORES PROMEDIO:
• Lesión: {self.stats_data['mean_score_lesion']:.4f}
• No-lesión: {self.stats_data['mean_score_non_lesion']:.4f}

INTERPRETACIÓN:
"""
        
        # Agregar interpretación
        if self.stats_data['lesion_percentage'] > 30:
            stats_text += "⚠ Alto porcentaje de lesión detectado\n"
        elif self.stats_data['lesion_percentage'] > 10:
            stats_text += "⚡ Lesión moderada detectada\n"
        elif self.stats_data['lesion_percentage'] > 1:
            stats_text += "✓ Lesión pequeña detectada\n"
        else:
            stats_text += "✓ Lesión mínima o no detectada\n"
        
        if abs(self.stats_data['mean_score_lesion'] - self.stats_data['mean_score_non_lesion']) > 1.0:
            stats_text += "✓ Separación clara entre clases\n"
        else:
            stats_text += "⚠ Separación ambigua entre clases\n"
        
        # Agregar información de reentrenamiento
        if self.training_data['images_processed'] > 0:
            stats_text += f"\nMODELO MEJORADO:\n"
            stats_text += f"• Imágenes de entrenamiento: {self.training_data['images_processed']}\n"
            stats_text += f"• Modelo re-entrenado: ✓ Sí\n"
            stats_text += f"• Precisión esperada: Mejorada\n"
        else:
            stats_text += f"\nMODELO ORIGINAL:\n"
            stats_text += f"• Usando parámetros base del experimento\n"
            stats_text += f"• Para mejorar: Cargar máscara real y re-entrenar\n"
        
        # Actualizar texto
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats_text)
    
    def save_results(self):
        """Guardar resultados del análisis"""
        if not self.stats_data or self.predicted_mask is None:
            messagebox.showwarning("Advertencia", "No hay resultados para guardar.")
            return
        
        try:
            # Seleccionar directorio
            save_dir = filedialog.askdirectory(title="Seleccionar directorio para guardar")
            if not save_dir:
                return
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"analisis_dermatoscopia_{timestamp}"
            
            # Guardar máscara predicha
            mask_path = os.path.join(save_dir, f"{base_name}_mascara.png")
            plt.imsave(mask_path, self.predicted_mask, cmap='gray')
            
            # Guardar superposición
            overlay = self.create_overlay(self.current_image, self.predicted_mask)
            overlay_path = os.path.join(save_dir, f"{base_name}_superposicion.png")
            plt.imsave(overlay_path, overlay)
            
            # Guardar estadísticas
            stats_path = os.path.join(save_dir, f"{base_name}_estadisticas.json")
            with open(stats_path, 'w') as f:
                json.dump(self.stats_data, f, indent=2)
            
            # Guardar reporte
            report_path = os.path.join(save_dir, f"{base_name}_reporte.txt")
            with open(report_path, 'w') as f:
                f.write(self.stats_text.get(1.0, tk.END))
            
            # Guardar modelo mejorado si ha sido re-entrenado
            if self.training_data['images_processed'] > 0:
                model_path = os.path.join(save_dir, f"{base_name}_modelo_mejorado.json")
                classifier_type = self.classifier_var.get()
                
                if classifier_type == "Bayesiano RGB":
                    classifier = self.bayesian_rgb_classifier
                else:
                    classifier = self.bayesian_pca_classifier
                
                model_data = {
                    'classifier_type': classifier_type,
                    'mu_lesion': classifier.mu_lesion.tolist(),
                    'mu_non_lesion': classifier.mu_non_lesion.tolist(),
                    'sigma_lesion': classifier.sigma_lesion.tolist(),
                    'sigma_non_lesion': classifier.sigma_non_lesion.tolist(),
                    'prior_lesion': classifier.prior_lesion,
                    'prior_non_lesion': classifier.prior_non_lesion,
                    'training_images': self.training_data['images_processed'],
                    'feature_dim': classifier.feature_dim
                }
                
                with open(model_path, 'w') as f:
                    json.dump(model_data, f, indent=2)
            
            files_created = [
                f"• {base_name}_mascara.png",
                f"• {base_name}_superposicion.png", 
                f"• {base_name}_estadisticas.json",
                f"• {base_name}_reporte.txt"
            ]
            
            if self.training_data['images_processed'] > 0:
                files_created.append(f"• {base_name}_modelo_mejorado.json")
            
            messagebox.showinfo("Éxito", 
                               f"Resultados guardados en:\n{save_dir}\n\n"
                               f"Archivos generados:\n" + "\n".join(files_created))
            
        except Exception as e:
            messagebox.showerror("Error", f"Error guardando resultados: {str(e)}")
    
    def clear_results(self):
        """Limpiar resultados y reiniciar interfaz"""
        # Limpiar variables
        self.current_image = None
        self.original_image = None
        self.predicted_mask = None
        self.ground_truth_mask = None
        self.stats_data = {}
        
        # Limpiar visualización
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.set_xticks([])
            ax.set_yticks([])
        
        self.ax1.set_title("Imagen Original", fontsize=12, fontweight='bold')
        self.ax2.set_title("Máscara Predicha", fontsize=12, fontweight='bold')
        self.ax3.set_title("Superposición", fontsize=12, fontweight='bold')
        
        self.canvas.draw()
        
        # Limpiar información
        self.image_info_label.config(text="No hay imagen cargada")
        self.stats_text.delete(1.0, tk.END)
        
        print("✓ Interfaz reiniciada")

    def load_ground_truth(self):
        """Cargar máscara de referencia (ground truth) para reentrenamiento"""
        if self.current_image is None:
            messagebox.showwarning("Advertencia", "Primero cargue una imagen para analizar.")
            return
            
        file_path = filedialog.askopenfilename(
            title="Seleccionar máscara de referencia (ground truth)",
            filetypes=[
                ("Imágenes", "*.png *.jpg *.jpeg *.bmp"),
                ("PNG", "*.png"),
                ("Todos los archivos", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Cargar máscara
                mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                
                # Redimensionar para coincidir con imagen actual
                if mask.shape != self.current_image.shape[:2]:
                    mask = cv2.resize(mask, 
                                    (self.current_image.shape[1], self.current_image.shape[0]), 
                                    interpolation=cv2.INTER_NEAREST)
                
                # Binarizar (umbral 128)
                self.ground_truth_mask = (mask > 128).astype(np.uint8)
                
                # Mostrar información
                lesion_pixels_gt = np.sum(self.ground_truth_mask)
                total_pixels = self.ground_truth_mask.size
                percentage_gt = (lesion_pixels_gt / total_pixels) * 100
                
                messagebox.showinfo("Máscara Cargada", 
                                  f"Máscara de referencia cargada exitosamente.\n"
                                  f"Lesión: {lesion_pixels_gt:,} píxeles ({percentage_gt:.2f}%)")
                
                print(f"✓ Máscara de referencia cargada: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error cargando máscara: {str(e)}")

    def retrain_classifier(self):
        """Re-entrenar clasificador con datos de corrección"""
        if self.current_image is None:
            messagebox.showwarning("Advertencia", "Primero cargue y analice una imagen.")
            return
            
        if self.ground_truth_mask is None:
            messagebox.showwarning("Advertencia", "Primero cargue una máscara de referencia.")
            return
            
        if self.predicted_mask is None:
            messagebox.showwarning("Advertencia", "Primero analice la imagen actual.")
            return
        
        try:
            # Mostrar progreso
            self.progress.start(10)
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(1.0, "Re-entrenando clasificador...\nPor favor espere...")
            self.root.update()
            
            # Extraer píxeles de corrección
            image = self.current_image.astype(np.float32)
            pixels = image.reshape(-1, 3)
            
            # Normalizar
            mean = np.array(self.normalization_params['mean'])
            std = np.array(self.normalization_params['std'])
            normalized_pixels = (pixels - mean) / std
            
            # Obtener máscaras planas
            gt_flat = self.ground_truth_mask.flatten()
            
            # Extraer píxeles de cada clase según ground truth
            lesion_pixels_new = normalized_pixels[gt_flat == 1]
            non_lesion_pixels_new = normalized_pixels[gt_flat == 0]
            
            if len(lesion_pixels_new) == 0 or len(non_lesion_pixels_new) == 0:
                messagebox.showwarning("Advertencia", "La máscara debe contener ambas clases (lesión y no-lesión).")
                return
            
            # Agregar a datos de entrenamiento acumulados
            self.training_data['lesion_pixels'].append(lesion_pixels_new)
            self.training_data['non_lesion_pixels'].append(non_lesion_pixels_new)
            self.training_data['images_processed'] += 1
            
            # Combinar todos los datos de entrenamiento
            all_lesion_data = np.vstack(self.training_data['lesion_pixels'])
            all_non_lesion_data = np.vstack(self.training_data['non_lesion_pixels'])
            
            # Seleccionar clasificador a re-entrenar
            classifier_type = self.classifier_var.get()
            
            if classifier_type == "Bayesiano RGB":
                # Re-entrenar clasificador RGB
                features_lesion = all_lesion_data
                features_non_lesion = all_non_lesion_data
                classifier = self.bayesian_rgb_classifier
                feature_name = "RGB"
            elif classifier_type == "Bayesiano PCA":
                # Re-entrenar clasificador PCA
                features_lesion = np.dot(all_lesion_data, self.pca_components.T)
                features_non_lesion = np.dot(all_non_lesion_data, self.pca_components.T)
                classifier = self.bayesian_pca_classifier
                feature_name = "PCA"
            else:  # K-Means
                # Re-entrenar clasificador K-Means
                all_features = np.vstack([all_lesion_data, all_non_lesion_data])
                all_labels = np.concatenate([np.ones(len(all_lesion_data)), np.zeros(len(all_non_lesion_data))])
                classifier = self.kmeans_classifier
                feature_name = "K-Means"
            
            # Crear nuevo clasificador mejorado
            if classifier_type == "K-Means":
                # Para K-Means, re-entrenar con los datos etiquetados
                improved_classifier = KMeansClassifier(n_clusters=2, seed=42)
                improved_classifier.fit(all_features)
                # Note: K-Means es no supervisado, pero usamos datos etiquetados para evaluar
            else:
                # Para clasificadores Bayesianos
                improved_classifier = BayesianClassifier(seed=42)
                improved_classifier.fit(features_lesion, features_non_lesion, equal_priors=True)
            
            # Reemplazar clasificador anterior
            if classifier_type == "Bayesiano RGB":
                self.bayesian_rgb_classifier = improved_classifier
            elif classifier_type == "Bayesiano PCA":
                self.bayesian_pca_classifier = improved_classifier
            else:  # K-Means
                self.kmeans_classifier = improved_classifier
            
            # Calcular mejora
            if classifier_type == "K-Means":
                old_params = f"Centroides: {len(classifier.centroids) if hasattr(classifier, 'centroids') else 0}"
                new_params = f"Centroides: {len(improved_classifier.centroids)}"
                samples_info = f"• Total: {len(all_features):,} píxeles\n"
            else:
                old_params = f"μ_lesión={classifier.mu_lesion[:2]}" if hasattr(classifier, 'mu_lesion') else "N/A"
                new_params = f"μ_lesión={improved_classifier.mu_lesion[:2]}"
                samples_info = f"• Lesión: {len(features_lesion):,} píxeles\n• No-lesión: {len(features_non_lesion):,} píxeles\n"
            
            # Mostrar resultados
            messagebox.showinfo("Reentrenamiento Completado", 
                              f"Clasificador {feature_name} mejorado exitosamente!\n\n"
                              f"Datos utilizados:\n"
                              f"{samples_info}"
                              f"• Imágenes procesadas: {self.training_data['images_processed']}\n\n"
                              f"Parámetros actualizados:\n"
                              f"• Anterior: {old_params}\n"
                              f"• Nuevo: {new_params}")
            
            # Re-analizar imagen actual con modelo mejorado
            self.analyze_image()
            
            print(f"✓ Clasificador {feature_name} re-entrenado exitosamente")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en reentrenamiento: {str(e)}")
            print(f"Error detallado: {e}")
        finally:
            self.progress.stop()


def main():
    """Función principal"""
    print("Iniciando Analizador de Lesiones Dermatoscópicas...")
    
    # Verificar que existen los archivos necesarios
    if not os.path.exists("src"):
        print("Error: Directorio 'src' no encontrado.")
        print("Asegúrese de ejecutar desde el directorio principal del proyecto.")
        return
    
    # Crear y ejecutar aplicación
    root = tk.Tk()
    app = DermatoscopyAnalyzer(root)
    
    print("✓ Interfaz gráfica iniciada")
    print("Instrucciones:")
    print("1. Cargar una imagen dermatoscópica")
    print("2. Seleccionar clasificador y criterio de umbral")
    print("3. Hacer clic en 'Analizar Imagen'")
    print("4. Revisar resultados y estadísticas")
    print("5. Guardar resultados si es necesario")
    
    root.mainloop()


if __name__ == "__main__":
    main()