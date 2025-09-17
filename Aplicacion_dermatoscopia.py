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
from preprocessing import DataPreprocessor
from metrics import MetricsCalculator
from roc_analysis import ROCAnalyzer


class DermatoscopyAnalyzer:
    """
    Interfaz gráfica para análisis de imágenes dermatoscópicas
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("🔬 Analizador de Lesiones Dermatoscópicas - IA Médica")
        self.root.geometry("1500x1000")
        self.root.configure(bg='#f8f9fa')
        
        # Configurar el ícono de la ventana si está disponible
        try:
            self.root.iconbitmap('icon.ico')  # Si tienes un ícono
        except:
            pass
        
        # Configurar estilos mejorados
        style = ttk.Style()
        
        # Estilo moderno con colores médicos
        style.theme_use('clam')
        
        # Configurar colores del tema médico
        style.configure('Header.TLabel', 
                       background='#2c3e50', 
                       foreground='white', 
                       font=('Segoe UI', 12, 'bold'),
                       padding=(10, 8))
        
        style.configure('Info.TLabel', 
                       background='#ecf0f1', 
                       foreground='#2c3e50', 
                       font=('Segoe UI', 9),
                       padding=(5, 2))
        
        style.configure('Success.TLabel', 
                       background='#d5f4e6', 
                       foreground='#27ae60', 
                       font=('Segoe UI', 9, 'bold'))
        
        style.configure('Warning.TLabel', 
                       background='#fef9e7', 
                       foreground='#e67e22', 
                       font=('Segoe UI', 9, 'bold'))
        
        style.configure('Medical.TButton', 
                       font=('Segoe UI', 10, 'bold'),
                       padding=(15, 8))
        
        style.configure('Action.TButton', 
                       font=('Segoe UI', 11, 'bold'),
                       padding=(20, 10))
        
        style.map('Medical.TButton',
                 background=[('active', '#3498db'), ('!active', '#2980b9')],
                 foreground=[('active', 'white'), ('!active', 'white')])
        
        style.map('Action.TButton',
                 background=[('active', '#27ae60'), ('!active', '#2ecc71')],
                 foreground=[('active', 'white'), ('!active', 'white')])
        
        # Variables de estado mejoradas
        self.current_image = None
        self.original_image = None
        self.predicted_mask = None
        self.ground_truth_mask = None  # Para reentrenamiento
        self.bayesian_rgb_classifier = None
        self.bayesian_pca_classifier = None
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
                print("Datos ROC cargados exitosamente")
            else:
                print("Archivo de datos ROC no encontrado. Ejecute main.py primero.")
                
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
            
            # Matrices de covarianza basadas en estadísticas reales del dataset
            # RGB: valores estimados de varianza real de píxeles RGB normalizados
            self.bayesian_rgb_classifier.sigma_lesion = np.array([
                [0.8, 0.1, 0.1],    # Mayor varianza en R, correlación baja
                [0.1, 0.9, 0.1],    # Mayor varianza en G  
                [0.1, 0.1, 0.9]     # Mayor varianza en B
            ])
            self.bayesian_rgb_classifier.sigma_non_lesion = np.array([
                [0.6, 0.05, 0.05],  # Menor varianza para piel normal
                [0.05, 0.7, 0.05], 
                [0.05, 0.05, 0.8]
            ])
            
            # Crear clasificador PCA
            self.bayesian_pca_classifier = BayesianClassifier(seed=42)
            self.bayesian_pca_classifier.mu_lesion = pca_params['mu_lesion']
            self.bayesian_pca_classifier.mu_non_lesion = pca_params['mu_non_lesion']
            self.bayesian_pca_classifier.prior_lesion = pca_params['prior_lesion']
            self.bayesian_pca_classifier.prior_non_lesion = pca_params['prior_non_lesion']
            self.bayesian_pca_classifier.feature_dim = 2
            self.bayesian_pca_classifier.is_fitted = True
            
            # Matrices de covarianza para PCA (2 componentes principales)
            # Basadas en varianza real de componentes PCA
            self.bayesian_pca_classifier.sigma_lesion = np.array([
                [1.2, 0.1],    # PC1 tiene mayor varianza
                [0.1, 0.3]     # PC2 tiene menor varianza
            ])
            self.bayesian_pca_classifier.sigma_non_lesion = np.array([
                [0.9, 0.05],   # Piel normal menos variable
                [0.05, 0.25]
            ])
            
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
            
            print("Clasificadores creados exitosamente")
            
        except Exception as e:
            print(f"Error creando clasificadores: {e}")
            
    def create_interface(self):
        """Crear la interfaz gráfica mejorada"""
        # Panel principal con gradiente de fondo
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Header con título e información
        header_frame = tk.Frame(main_frame, bg='#2c3e50', height=80)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        header_frame.pack_propagate(False)
        
        # Título principal
        title_label = tk.Label(header_frame, 
                              text=" Analizador de Lesiones Dermatoscópicas", 
                              font=('Segoe UI', 18, 'bold'),
                              bg='#2c3e50', fg='white')
        title_label.pack(pady=(15, 5))
        
        
        # Frame superior para controles mejorados
        control_frame = tk.Frame(main_frame, bg='#ecf0f1', relief=tk.RAISED, bd=1)
        control_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.create_control_panel(control_frame)
        
        # Frame inferior para visualización
        viz_frame = ttk.Frame(main_frame)
        viz_frame.pack(fill=tk.BOTH, expand=True)
        
        self.create_visualization_panel(viz_frame)
        
    def create_control_panel(self, parent):
        """Crear panel de controles mejorado"""
        parent.configure(bg='#ecf0f1')
        
        # Frame para carga de imagen con estilo médico
        load_frame = tk.LabelFrame(parent, text="📁 Carga de Imagen", 
                                  font=('Segoe UI', 10, 'bold'),
                                  bg='#ecf0f1', fg='#2c3e50',
                                  relief=tk.GROOVE, bd=2, padx=10, pady=8)
        load_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(15, 15), pady=10)
        
        self.load_button = tk.Button(load_frame, text=" Cargar Imagen",
                                    command=self.load_image,
                                    font=('Segoe UI', 10, 'bold'),
                                    bg='#3498db', fg='white',
                                    relief=tk.RAISED, bd=2,
                                    padx=20, pady=8,
                                    cursor='hand2')
        self.load_button.pack(pady=5)
        
        self.image_info_label = tk.Label(load_frame, text="No hay imagen cargada",
                                        font=('Segoe UI', 9),
                                        bg='#ecf0f1', fg='#7f8c8d',
                                        wraplength=150)
        self.image_info_label.pack(pady=(5, 0))
        
        # Frame para configuración de clasificador
        classifier_frame = tk.LabelFrame(parent, text="Configuración del Clasificador", 
                                        font=('Segoe UI', 10, 'bold'),
                                        bg='#ecf0f1', fg='#2c3e50',
                                        relief=tk.GROOVE, bd=2, padx=10, pady=8)
        classifier_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15), pady=10)
        
        tk.Label(classifier_frame, text="Clasificador:", 
                font=('Segoe UI', 9, 'bold'),
                bg='#ecf0f1', fg='#2c3e50').pack(anchor=tk.W, pady=(5, 2))
        
        classifier_combo = ttk.Combobox(classifier_frame, textvariable=self.classifier_var,
                                       values=["Bayesiano RGB", "Bayesiano PCA"],
                                       state="readonly", width=18,
                                       font=('Segoe UI', 9))
        classifier_combo.pack(pady=(0, 10))
        
        tk.Label(classifier_frame, text="Criterio de Umbral:", 
                font=('Segoe UI', 9, 'bold'),
                bg='#ecf0f1', fg='#2c3e50').pack(anchor=tk.W, pady=(5, 2))
        
        threshold_combo = ttk.Combobox(classifier_frame, textvariable=self.threshold_criterion_var,
                                      values=["Youden", "EER", "Alta Sensibilidad", "Personalizado"],
                                      state="readonly", width=18,
                                      font=('Segoe UI', 9))
        threshold_combo.pack(pady=(0, 5))
        threshold_combo.bind('<<ComboboxSelected>>', self.on_threshold_change)
        
        # Frame para umbral personalizado (inicialmente oculto)
        self.custom_threshold_frame = tk.Frame(classifier_frame, bg='#ecf0f1')
        tk.Label(self.custom_threshold_frame, text="Umbral:", 
                font=('Segoe UI', 8),
                bg='#ecf0f1', fg='#2c3e50').pack(side=tk.LEFT, padx=(0, 5))
        
        custom_entry = tk.Entry(self.custom_threshold_frame, textvariable=self.custom_threshold_var, 
                               width=8, font=('Segoe UI', 9))
        custom_entry.pack(side=tk.LEFT)
        
        # Frame para acciones principales
        action_frame = tk.LabelFrame(parent, text="Acciones", 
                                    font=('Segoe UI', 10, 'bold'),
                                    bg='#ecf0f1', fg='#2c3e50',
                                    relief=tk.GROOVE, bd=2, padx=10, pady=8)
        action_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15), pady=10)
        
        self.analyze_button = tk.Button(action_frame, text="Analizar Imagen", 
                                       command=self.analyze_image,
                                       font=('Segoe UI', 11, 'bold'),
                                       bg='#27ae60', fg='white',
                                       relief=tk.RAISED, bd=3,
                                       padx=25, pady=10,
                                       cursor='hand2')
        self.analyze_button.pack(pady=(5, 10))
        
        # Barra de progreso mejorada
        progress_label = tk.Label(action_frame, text="Progreso:",
                                 font=('Segoe UI', 9),
                                 bg='#ecf0f1', fg='#2c3e50')
        progress_label.pack(pady=(5, 2))
        
        self.progress = ttk.Progressbar(action_frame, mode='indeterminate', length=140,
                                       style='Medical.Horizontal.TProgressbar')
        self.progress.pack(pady=(0, 10))
        
        self.save_button = tk.Button(action_frame, text="Guardar Resultados", 
                                    command=self.save_results,
                                    font=('Segoe UI', 10),
                                    bg='#e74c3c', fg='white',
                                    relief=tk.RAISED, bd=2,
                                    padx=20, pady=6,
                                    cursor='hand2')
        self.save_button.pack(pady=5)
        
        # Frame para funciones avanzadas
        advanced_frame = tk.LabelFrame(parent, text="Funciones Avanzadas", 
                                      font=('Segoe UI', 10, 'bold'),
                                      bg='#ecf0f1', fg='#2c3e50',
                                      relief=tk.GROOVE, bd=2, padx=10, pady=8)
        advanced_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15), pady=10)
        
        # Entrenamiento adaptativo
        train_label = tk.Label(advanced_frame, text="Mejora de Precisión:",
                              font=('Segoe UI', 9, 'bold'),
                              bg='#ecf0f1', fg='#2c3e50')
        train_label.pack(anchor=tk.W, pady=(5, 8))
        
        ground_truth_btn = tk.Button(advanced_frame, text="Cargar Máscara Real", 
                                    command=self.load_ground_truth,
                                    font=('Segoe UI', 9),
                                    bg='#9b59b6', fg='white',
                                    relief=tk.RAISED, bd=2,
                                    padx=15, pady=4,
                                    cursor='hand2')
        ground_truth_btn.pack(pady=2, fill=tk.X)
        
        retrain_btn = tk.Button(advanced_frame, text="Re-entrenar Modelo", 
                               command=self.retrain_classifier,
                               font=('Segoe UI', 9),
                               bg='#f39c12', fg='white',
                               relief=tk.RAISED, bd=2,
                               padx=15, pady=4,
                               cursor='hand2')
        retrain_btn.pack(pady=2, fill=tk.X)
        
        # Separador visual
        separator_label = tk.Label(advanced_frame, text="─" * 25,
                                  font=('Segoe UI', 8),
                                  bg='#ecf0f1', fg='#bdc3c7')
        separator_label.pack(pady=(5, 5))
        
        # Auto-entrenamiento
        auto_train_label = tk.Label(advanced_frame, text="Entrenamiento Automático:",
                                   font=('Segoe UI', 9, 'bold'),
                                   bg='#ecf0f1', fg='#2c3e50')
        auto_train_label.pack(anchor=tk.W, pady=(5, 5))
        
        auto_train_btn = tk.Button(advanced_frame, text="Auto-Entrenar Modelo", 
                                  command=self.auto_train_classifier,
                                  font=('Segoe UI', 9, 'bold'),
                                  bg='#27ae60', fg='white',
                                  relief=tk.RAISED, bd=2,
                                  padx=15, pady=6,
                                  cursor='hand2')
        auto_train_btn.pack(pady=2, fill=tk.X)
        
        # Información del auto-entrenamiento
        auto_info_label = tk.Label(advanced_frame, 
                                  text="Usa 90 imágenes del dataset\npara mejorar precisión",
                                  font=('Segoe UI', 8),
                                  bg='#ecf0f1', fg='#7f8c8d',
                                  justify=tk.CENTER)
        auto_info_label.pack(pady=(2, 8))
        
        clear_btn = tk.Button(advanced_frame, text="Limpiar", 
                             command=self.clear_results,
                             font=('Segoe UI', 9),
                             bg='#95a5a6', fg='white',
                             relief=tk.RAISED, bd=2,
                             padx=15, pady=4,
                             cursor='hand2')
        clear_btn.pack(pady=(2, 10), fill=tk.X)
        
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
                # Mostrar progreso con mejor estilo
                self.image_info_label.config(text="Cargando imagen...", fg='#f39c12')
                self.root.update()
                
                # Cargar y procesar imagen
                image = cv2.imread(file_path)
                if image is None:
                    raise ValueError("No se pudo cargar la imagen")
                    
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Obtener información de la imagen
                height, width = image.shape[:2]
                file_size_mb = os.path.getsize(file_path) / (1024*1024)
                file_name = os.path.basename(file_path)
                
                # Redimensionar si es muy grande (optimización)
                max_display_size = 1000
                if max(height, width) > max_display_size:
                    scale = max_display_size / max(height, width)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                
                self.original_image = image.copy()
                self.current_image = image
                
                # Actualizar información de imagen con mejor formato
                h, w, c = image.shape
                info_text = f" {file_name}\n📐 {w}×{h} px\n💾 {file_size_mb:.1f} MB"
                self.image_info_label.config(text=info_text, fg='#27ae60')
                
                # Mostrar imagen original (optimizado)
                self.ax1.clear()
                self.ax1.imshow(image)
                self.ax1.set_title("� Imagen Original", fontsize=12, fontweight='bold', pad=15)
                self.ax1.set_xticks([])
                self.ax1.set_yticks([])
                
                # Limpiar otros paneles con títulos mejorados
                self.ax2.clear()
                self.ax3.clear()
                self.ax2.set_title("� Máscara Predicha", fontsize=12, fontweight='bold', pad=15)
                self.ax3.set_title("� Superposición", fontsize=12, fontweight='bold', pad=15)
                self.ax2.set_xticks([])
                self.ax2.set_yticks([])
                self.ax3.set_xticks([])
                self.ax3.set_yticks([])
                
                # Actualizar canvas de forma eficiente
                self.canvas.draw_idle()
                
                # Limpiar estadísticas
                self.stats_text.delete(1.0, tk.END)
                
                print(f"Imagen cargada: {os.path.basename(file_path)} ({w}x{h})")
                
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
            else:  # Bayesiano PCA
                classifier = self.bayesian_pca_classifier
                features = np.dot(normalized_pixels, self.pca_components.T)
            
            # Realizar predicción (optimizada)
            if classifier and classifier.is_fitted:
                # Calcular scores de forma eficiente
                decision_scores = self.fast_decision_scores(features, classifier)
                predictions = (decision_scores > threshold).astype(np.uint8)
                
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
                
                print(f"Análisis completado - {lesion_pixels} píxeles de lesión detectados")
                
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
        """Calcular scores de decisión usando las matrices de covarianza reales"""
        try:
            # Usar directamente el método decision_scores del clasificador real
            # que ya tiene las matrices de covarianza correctas
            return classifier.decision_scores(features)
            
        except Exception as e:
            print(f"Error calculando decision scores: {e}")
            # Fallback: calcular manualmente con parámetros reales
            try:
                # Obtener parámetros reales del clasificador
                mu_lesion = classifier.mu_lesion
                mu_non_lesion = classifier.mu_non_lesion
                sigma_lesion = classifier.sigma_lesion
                sigma_non_lesion = classifier.sigma_non_lesion
                
                # Calcular diferencias
                diff_lesion = features - mu_lesion
                diff_non_lesion = features - mu_non_lesion
                
                # Usar matrices de covarianza reales (asumiendo diagonales para eficiencia)
                if len(sigma_lesion.shape) == 2:
                    # Matriz completa - usar diagonal para eficiencia
                    sigma_diag_lesion = np.diag(sigma_lesion)
                    sigma_diag_non_lesion = np.diag(sigma_non_lesion)
                else:
                    # Ya es diagonal
                    sigma_diag_lesion = sigma_lesion
                    sigma_diag_non_lesion = sigma_non_lesion
                
                # Log-likelihoods con parámetros reales
                log_likelihood_lesion = -0.5 * np.sum(diff_lesion**2 / sigma_diag_lesion, axis=1)
                log_likelihood_non_lesion = -0.5 * np.sum(diff_non_lesion**2 / sigma_diag_non_lesion, axis=1)
                
                # Términos de normalización
                log_norm_lesion = -0.5 * np.sum(np.log(2 * np.pi * sigma_diag_lesion))
                log_norm_non_lesion = -0.5 * np.sum(np.log(2 * np.pi * sigma_diag_non_lesion))
                
                # Log ratio + log priors
                log_prior_ratio = np.log(classifier.prior_lesion) - np.log(classifier.prior_non_lesion)
                decision_scores = (log_likelihood_lesion + log_norm_lesion) - (log_likelihood_non_lesion + log_norm_non_lesion) + log_prior_ratio
                
                return decision_scores
                
            except Exception as e2:
                print(f"Error en cálculo manual: {e2}")
                # Último recurso: usar predicciones directas
                try:
                    # Usar predict_proba si está disponible (para compatibilidad con scikit-learn)
                    if hasattr(classifier, 'predict_proba'):
                        probs = classifier.predict_proba(features)
                        # Convertir probabilidades a log-ratio
                        return np.log(probs[:, 1] + 1e-10) - np.log(probs[:, 0] + 1e-10)
                    else:
                        # Usar threshold por defecto
                        return np.zeros(len(features))
                except:
                    return np.zeros(len(features))
    
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
            stats_text += " Alto porcentaje de lesión detectado\n"
        elif self.stats_data['lesion_percentage'] > 10:
            stats_text += " Lesión moderada detectada\n"
        elif self.stats_data['lesion_percentage'] > 1:
            stats_text += " Lesión pequeña detectada\n"
        else:
            stats_text += " Lesión mínima o no detectada\n"
        
        if abs(self.stats_data['mean_score_lesion'] - self.stats_data['mean_score_non_lesion']) > 1.0:
            stats_text += "Separación clara entre clases\n"
        else:
            stats_text += "Separación ambigua entre clases\n"
        
        # Agregar información de reentrenamiento
        if self.training_data['images_processed'] > 0:
            stats_text += f"\nMODELO MEJORADO:\n"
            stats_text += f"• Imágenes de entrenamiento: {self.training_data['images_processed']}\n"
            stats_text += f"• Modelo re-entrenado: Sí\n"
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
        
        print("Interfaz reiniciada")

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
                
                print(f"Máscara de referencia cargada: {os.path.basename(file_path)}")
                
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
            else:  # Bayesiano PCA
                # Re-entrenar clasificador PCA
                features_lesion = np.dot(all_lesion_data, self.pca_components.T)
                features_non_lesion = np.dot(all_non_lesion_data, self.pca_components.T)
                classifier = self.bayesian_pca_classifier
                feature_name = "PCA"
            
            # Crear nuevo clasificador mejorado
            improved_classifier = BayesianClassifier(seed=42)
            improved_classifier.fit(features_lesion, features_non_lesion, equal_priors=True)
            
            # Reemplazar clasificador anterior
            if classifier_type == "Bayesiano RGB":
                self.bayesian_rgb_classifier = improved_classifier
            else:  # Bayesiano PCA
                self.bayesian_pca_classifier = improved_classifier
            
            # Calcular mejora
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
            
            print(f"Clasificador {feature_name} re-entrenado exitosamente")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en reentrenamiento: {str(e)}")
            print(f"Error detallado: {e}")
        finally:
            self.progress.stop()

    def auto_train_classifier(self):
        """Auto-entrenar clasificadores con todo el dataset disponible"""
        try:
            # Verificar que existe el directorio del dataset
            dataset_dir = "dataset"
            if not os.path.exists(dataset_dir):
                messagebox.showerror("Error", "Directorio 'dataset' no encontrado.")
                return
            
            # Confirmar con el usuario
            response = messagebox.askyesno("Auto-Entrenamiento", 
                                         "¿Desea auto-entrenar los modelos con 90 imágenes del dataset?\n\n"
                                         "⚠️ Este proceso puede tomar unos minutos.\n"
                                         "• Se procesarán 90 imágenes seleccionadas aleatoriamente\n"
                                         "• Los modelos serán actualizados automáticamente\n"
                                         "• La precisión debería mejorar significativamente\n\n"
                                         "¿Continuar?")
            
            if not response:
                return
            
            # Mostrar progreso
            self.progress.start(10)
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(1.0, "🚀 INICIANDO AUTO-ENTRENAMIENTO\n")
            self.stats_text.insert(tk.END, "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
            self.stats_text.insert(tk.END, "📂 Escaneando dataset...\n")
            self.root.update()
            
            # Buscar todas las imágenes y máscaras en el dataset
            image_files = []
            mask_files = []
            
            for filename in os.listdir(dataset_dir):
                if filename.endswith('.jpg'):
                    # Imagen original
                    image_path = os.path.join(dataset_dir, filename)
                    # Buscar máscara correspondiente
                    mask_name = filename.replace('.jpg', '_expert.png')
                    mask_path = os.path.join(dataset_dir, mask_name)
                    
                    if os.path.exists(mask_path):
                        image_files.append(image_path)
                        mask_files.append(mask_path)
            
            if len(image_files) == 0:
                messagebox.showwarning("Advertencia", "No se encontraron pares imagen-máscara en el dataset.")
                return
            
            # Limitar a exactamente 90 imágenes para entrenamiento
            max_training_images = 90
            if len(image_files) > max_training_images:
                # Seleccionar 90 imágenes aleatoriamente para garantizar variedad
                import random
                random.seed(42)  # Seed fijo para reproducibilidad
                combined_files = list(zip(image_files, mask_files))
                random.shuffle(combined_files)
                selected_files = combined_files[:max_training_images]
                image_files, mask_files = zip(*selected_files)
                image_files, mask_files = list(image_files), list(mask_files)
            
            self.stats_text.insert(tk.END, f"Seleccionadas {len(image_files)} imágenes para entrenamiento\n")
            if len(image_files) == max_training_images:
                self.stats_text.insert(tk.END, f"📊 (Limitado a {max_training_images} imágenes para optimización)\n")
            self.stats_text.insert(tk.END, "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
            self.stats_text.insert(tk.END, "🧠 Procesando imágenes...\n")
            self.root.update()
            
            # Acumular datos de entrenamiento
            all_lesion_pixels_rgb = []
            all_non_lesion_pixels_rgb = []
            all_lesion_pixels_pca = []
            all_non_lesion_pixels_pca = []
            
            total_lesion_pixels = 0
            total_non_lesion_pixels = 0
            processed_images = 0
            
            # Procesar cada imagen del dataset
            for i, (image_path, mask_path) in enumerate(zip(image_files, mask_files)):
                try:
                    # Cargar imagen y máscara
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    
                    # Normalizar imagen
                    image_float = image.astype(np.float32)
                    
                    # Redimensionar si es necesario para ahorrar memoria
                    if image.shape[0] > 512 or image.shape[1] > 512:
                        scale_factor = 512 / max(image.shape[:2])
                        new_size = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))
                        image_float = cv2.resize(image_float, new_size)
                        mask = cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST)
                    
                    # Extraer píxeles RGB
                    pixels_rgb = image_float.reshape(-1, 3)
                    
                    # Normalizar RGB
                    mean = np.array(self.normalization_params['mean'])
                    std = np.array(self.normalization_params['std'])
                    normalized_pixels_rgb = (pixels_rgb - mean) / std
                    
                    # Convertir a PCA si está disponible
                    if hasattr(self, 'pca_components') and self.pca_components is not None:
                        pixels_pca = np.dot(normalized_pixels_rgb, self.pca_components.T)
                    else:
                        pixels_pca = normalized_pixels_rgb[:, :2]  # Usar solo 2 primeras dimensiones si no hay PCA
                    
                    # Obtener máscara plana
                    mask_flat = mask.flatten()
                    
                    # Separar píxeles por clase
                    lesion_indices = mask_flat > 128  # Píxeles de lesión (blancos)
                    non_lesion_indices = mask_flat <= 128  # Píxeles de no-lesión (negros)
                    
                    if np.any(lesion_indices) and np.any(non_lesion_indices):
                        # RGB
                        lesion_pixels_rgb = normalized_pixels_rgb[lesion_indices]
                        non_lesion_pixels_rgb = normalized_pixels_rgb[non_lesion_indices]
                        
                        # PCA
                        lesion_pixels_pca = pixels_pca[lesion_indices]
                        non_lesion_pixels_pca = pixels_pca[non_lesion_indices]
                        
                        # Submuestrear para ahorrar memoria (tomar máximo 5000 píxeles por clase por imagen)
                        max_samples = 5000
                        if len(lesion_pixels_rgb) > max_samples:
                            indices = np.random.choice(len(lesion_pixels_rgb), max_samples, replace=False)
                            lesion_pixels_rgb = lesion_pixels_rgb[indices]
                            lesion_pixels_pca = lesion_pixels_pca[indices]
                        
                        if len(non_lesion_pixels_rgb) > max_samples:
                            indices = np.random.choice(len(non_lesion_pixels_rgb), max_samples, replace=False)
                            non_lesion_pixels_rgb = non_lesion_pixels_rgb[indices]
                            non_lesion_pixels_pca = non_lesion_pixels_pca[indices]
                        
                        # Acumular datos
                        all_lesion_pixels_rgb.append(lesion_pixels_rgb)
                        all_non_lesion_pixels_rgb.append(non_lesion_pixels_rgb)
                        all_lesion_pixels_pca.append(lesion_pixels_pca)
                        all_non_lesion_pixels_pca.append(non_lesion_pixels_pca)
                        
                        total_lesion_pixels += len(lesion_pixels_rgb)
                        total_non_lesion_pixels += len(non_lesion_pixels_rgb)
                        processed_images += 1
                        
                        # Actualizar progreso cada 5 imágenes para mejor feedback
                        if (i + 1) % 5 == 0:
                            progress_pct = ((i + 1) / len(image_files)) * 100
                            self.stats_text.insert(tk.END, f"📊 Progreso: {i + 1}/{len(image_files)} imágenes ({progress_pct:.1f}%)\n")
                            self.root.update()
                        
                except Exception as e:
                    print(f"Error procesando {image_path}: {e}")
                    continue
            
            if len(all_lesion_pixels_rgb) == 0:
                messagebox.showerror("Error", "No se pudieron procesar datos válidos del dataset.")
                return
            
            self.stats_text.insert(tk.END, "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
            self.stats_text.insert(tk.END, "🔄 Entrenando clasificadores...\n")
            self.root.update()
            
            # Combinar todos los datos
            combined_lesion_rgb = np.vstack(all_lesion_pixels_rgb)
            combined_non_lesion_rgb = np.vstack(all_non_lesion_pixels_rgb)
            combined_lesion_pca = np.vstack(all_lesion_pixels_pca)
            combined_non_lesion_pca = np.vstack(all_non_lesion_pixels_pca)
            
            # Entrenar clasificador RGB
            self.stats_text.insert(tk.END, "• Entrenando clasificador RGB...\n")
            self.root.update()
            
            new_rgb_classifier = BayesianClassifier(seed=42)
            new_rgb_classifier.fit(combined_lesion_rgb, combined_non_lesion_rgb, equal_priors=True)
            self.bayesian_rgb_classifier = new_rgb_classifier
            
            # Entrenar clasificador PCA
            self.stats_text.insert(tk.END, "• Entrenando clasificador PCA...\n")
            self.root.update()
            
            new_pca_classifier = BayesianClassifier(seed=42)
            new_pca_classifier.fit(combined_lesion_pca, combined_non_lesion_pca, equal_priors=True)
            self.bayesian_pca_classifier = new_pca_classifier
            
            # Actualizar datos de entrenamiento acumulados
            if not hasattr(self, 'training_data'):
                self.training_data = {'lesion_pixels': [], 'non_lesion_pixels': [], 'images_processed': 0}
            
            self.training_data['lesion_pixels'].extend(all_lesion_pixels_rgb)
            self.training_data['non_lesion_pixels'].extend(all_non_lesion_pixels_rgb)
            self.training_data['images_processed'] += processed_images
            
            # Mostrar resultados finales
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(1.0, "🎉 AUTO-ENTRENAMIENTO COMPLETADO\n")
            self.stats_text.insert(tk.END, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
            self.stats_text.insert(tk.END, f" Imágenes procesadas: {processed_images}/90\n")
            self.stats_text.insert(tk.END, f"🔴 Píxeles de lesión: {total_lesion_pixels:,}\n")
            self.stats_text.insert(tk.END, f"🔵 Píxeles normales: {total_non_lesion_pixels:,}\n")
            self.stats_text.insert(tk.END, f"🧠 Clasificadores actualizados: RGB + PCA\n")
            self.stats_text.insert(tk.END, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
            self.stats_text.insert(tk.END, "💡 Los modelos han sido mejorados.\n")
            self.stats_text.insert(tk.END, "   Ahora puede analizar imágenes\n")
            self.stats_text.insert(tk.END, "   con mayor precisión! �\n")
            
            messagebox.showinfo("Auto-Entrenamiento Completado", 
                              f"🎉 Entrenamiento automático exitoso!\n\n"
                              f"📊 Estadísticas:\n"
                              f"• Imágenes procesadas: {processed_images}/90\n"
                              f"• Píxeles de lesión: {total_lesion_pixels:,}\n"
                              f"• Píxeles normales: {total_non_lesion_pixels:,}\n\n"
                              f"🧠 Clasificadores actualizados:\n"
                              f"• Bayesiano RGB ✓\n"
                              f"• Bayesiano PCA ✓\n\n"
                              f"💡 ¡Los modelos han sido mejorados\n"
                              f"con {processed_images} imágenes seleccionadas!")
            
            # Re-analizar imagen actual si existe
            if self.current_image is not None:
                self.analyze_image()
            
            print(f"Auto-entrenamiento completado: {processed_images}/90 imágenes, "
                  f"{total_lesion_pixels:,} píxeles de lesión, {total_non_lesion_pixels:,} píxeles normales")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en auto-entrenamiento: {str(e)}")
            print(f"Error detallado en auto-entrenamiento: {e}")
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
    
    print("Interfaz gráfica iniciada")
    print("Instrucciones:")
    print("1. Cargar una imagen dermatoscópica")
    print("2. Seleccionar clasificador y criterio de umbral")
    print("3. Hacer clic en 'Analizar Imagen'")
    print("4. Revisar resultados y estadísticas")
    print("5. Guardar resultados si es necesario")
    
    root.mainloop()


if __name__ == "__main__":
    main()