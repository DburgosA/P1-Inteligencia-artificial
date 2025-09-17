"""
Módulo de visualización para el proyecto de segmentación de lesiones
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import os
from matplotlib.patches import Rectangle
import pandas as pd
import time

def show_plot_briefly(duration=1.0):
    """Muestra el plot actual por un tiempo específico y luego lo cierra"""
    plt.show(block=False)
    plt.pause(duration)
    plt.close()


class VisualizationManager:
    """
    Clase para manejar todas las visualizaciones del proyecto
    """
    
    def __init__(self, results_dir: str = "results"):
        """
        Inicializa el manejador de visualizaciones
        
        Args:
            results_dir: Directorio donde guardar las visualizaciones
        """
        self.results_dir = results_dir
        self.ensure_results_dir()
        
        # Configurar estilo de matplotlib
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Configuraciones de colores consistentes
        self.colors = {
            'lesion': '#FF6B6B',      # Rojo para lesión
            'non_lesion': '#4ECDC4',  # Verde-azul para no-lesión
            'bayesian_rgb': '#3498DB',     # Azul
            'bayesian_pca': '#E74C3C',     # Rojo
            'kmeans': '#2ECC71',           # Verde
            'reference': '#34495E',        # Gris oscuro
            'prediction': '#F39C12'        # Naranja
        }
    
    def ensure_results_dir(self):
        """Crea el directorio de resultados si no existe"""
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
    
    def plot_rgb_histograms(self, lesion_pixels: np.ndarray, non_lesion_pixels: np.ndarray,
                           title: str = "Distribuciones RGB por Clase", 
                           save_name: Optional[str] = None):
        """
        Genera histogramas RGB comparativos entre clases
        
        Args:
            lesion_pixels: Píxeles de lesión (N, 3)
            non_lesion_pixels: Píxeles de no-lesión (M, 3)
            title: Título de la figura
            save_name: Nombre del archivo para guardar
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        channels = ['R', 'G', 'B']
        channel_colors = ['red', 'green', 'blue']
        
        for i, (channel, color) in enumerate(zip(channels, channel_colors)):
            # Histograma superpuesto
            axes[0, i].hist(lesion_pixels[:, i], bins=50, alpha=0.7, 
                           color=self.colors['lesion'], label='Lesión', density=True)
            axes[0, i].hist(non_lesion_pixels[:, i], bins=50, alpha=0.7, 
                           color=self.colors['non_lesion'], label='No-lesión', density=True)
            axes[0, i].set_title(f'Canal {channel} - Comparación')
            axes[0, i].set_xlabel('Intensidad')
            axes[0, i].set_ylabel('Densidad')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)
            
            # Histogramas separados
            axes[1, i].hist(lesion_pixels[:, i], bins=50, alpha=0.8, 
                           color=self.colors['lesion'], density=True)
            axes[1, i].set_title(f'Canal {channel} - Solo Lesión')
            axes[1, i].set_xlabel('Intensidad')
            axes[1, i].set_ylabel('Densidad')
            axes[1, i].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_name:
            save_path = os.path.join(self.results_dir, f"{save_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Histogramas RGB guardados en: {save_path}")
        
        show_plot_briefly(1.0)
    
    def plot_dataset_statistics(self, stats: Dict, save_name: Optional[str] = None):
        """
        Visualiza estadísticas del dataset
        
        Args:
            stats: Diccionario con estadísticas por split
            save_name: Nombre del archivo para guardar
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Número de imágenes por split
        splits = list(stats.keys())
        n_images = [stats[split]['n_images'] for split in splits]
        
        axes[0, 0].bar(splits, n_images, color=[self.colors['bayesian_rgb'], 
                                               self.colors['bayesian_pca'], 
                                               self.colors['kmeans']])
        axes[0, 0].set_title('Número de Imágenes por Split')
        axes[0, 0].set_ylabel('Número de Imágenes')
        
        # 2. Proporción de píxeles por clase
        lesion_pixels = [stats[split]['n_lesion_pixels'] for split in splits]
        non_lesion_pixels = [stats[split]['n_non_lesion_pixels'] for split in splits]
        
        x = np.arange(len(splits))
        width = 0.35
        
        axes[0, 1].bar(x - width/2, lesion_pixels, width, label='Lesión', 
                      color=self.colors['lesion'])
        axes[0, 1].bar(x + width/2, non_lesion_pixels, width, label='No-lesión', 
                      color=self.colors['non_lesion'])
        axes[0, 1].set_title('Píxeles por Clase y Split')
        axes[0, 1].set_ylabel('Número de Píxeles')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(splits)
        axes[0, 1].legend()
        axes[0, 1].set_yscale('log')
        
        # 3. Medias RGB por clase (solo train)
        if 'train' in stats and 'lesion_rgb_mean' in stats['train']:
            train_stats = stats['train']
            lesion_means = train_stats['lesion_rgb_mean']
            non_lesion_means = train_stats['non_lesion_rgb_mean']
            
            channels = ['R', 'G', 'B']
            x = np.arange(len(channels))
            
            axes[1, 0].bar(x - width/2, lesion_means, width, label='Lesión', 
                          color=self.colors['lesion'])
            axes[1, 0].bar(x + width/2, non_lesion_means, width, label='No-lesión', 
                          color=self.colors['non_lesion'])
            axes[1, 0].set_title('Medias RGB por Canal (Entrenamiento)')
            axes[1, 0].set_ylabel('Intensidad Media')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(channels)
            axes[1, 0].legend()
        
        # 4. Desviaciones estándar RGB por clase (solo train)
        if 'train' in stats and 'lesion_rgb_std' in stats['train']:
            train_stats = stats['train']
            lesion_stds = train_stats['lesion_rgb_std']
            non_lesion_stds = train_stats['non_lesion_rgb_std']
            
            axes[1, 1].bar(x - width/2, lesion_stds, width, label='Lesión', 
                          color=self.colors['lesion'])
            axes[1, 1].bar(x + width/2, non_lesion_stds, width, label='No-lesión', 
                          color=self.colors['non_lesion'])
            axes[1, 1].set_title('Desviaciones Estándar RGB (Entrenamiento)')
            axes[1, 1].set_ylabel('Desviación Estándar')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(channels)
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_name:
            save_path = os.path.join(self.results_dir, f"{save_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Estadísticas del dataset guardadas en: {save_path}")
        
        show_plot_briefly(1.0)
    
    def plot_pca_analysis(self, pca_info: Dict, save_name: Optional[str] = None):
        """
        Visualiza análisis de PCA
        
        Args:
            pca_info: Información de PCA del preprocessor
            save_name: Nombre del archivo para guardar
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Varianza explicada por componente
        n_components = len(pca_info['explained_variance_ratio'])
        components = range(1, n_components + 1)
        
        axes[0, 0].bar(components, pca_info['explained_variance_ratio'], 
                      color=self.colors['bayesian_pca'])
        axes[0, 0].set_title('Varianza Explicada por Componente')
        axes[0, 0].set_xlabel('Componente Principal')
        axes[0, 0].set_ylabel('Proporción de Varianza')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Varianza acumulada
        cumsum_variance = np.cumsum(pca_info['explained_variance_ratio'])
        axes[0, 1].plot(components, cumsum_variance, 'o-', 
                       color=self.colors['bayesian_pca'], linewidth=2, markersize=6)
        axes[0, 1].axhline(y=0.95, color='red', linestyle='--', alpha=0.7, 
                          label='95% Varianza')
        axes[0, 1].set_title('Varianza Acumulada')
        axes[0, 1].set_xlabel('Número de Componentes')
        axes[0, 1].set_ylabel('Varianza Acumulada')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 1])
        
        # 3. Componentes principales (loadings)
        if n_components >= 2:
            components_matrix = pca_info['components']
            
            # Heatmap de las primeras componentes
            n_show = min(3, n_components)
            im = axes[1, 0].imshow(components_matrix[:n_show], cmap='RdBu_r', aspect='auto')
            axes[1, 0].set_title('Componentes Principales (Loadings)')
            axes[1, 0].set_xlabel('Características Originales (R, G, B)')
            axes[1, 0].set_ylabel('Componente Principal')
            axes[1, 0].set_xticks([0, 1, 2])
            axes[1, 0].set_xticklabels(['R', 'G', 'B'])
            axes[1, 0].set_yticks(range(n_show))
            axes[1, 0].set_yticklabels([f'PC{i+1}' for i in range(n_show)])
            plt.colorbar(im, ax=axes[1, 0])
        
        # 4. Contribución de cada canal RGB a las componentes
        if n_components >= 1:
            channels = ['R', 'G', 'B']
            pc1_contributions = np.abs(pca_info['components'][0])
            
            axes[1, 1].bar(channels, pc1_contributions, color=['red', 'green', 'blue'])
            axes[1, 1].set_title('Contribución de Canales RGB a PC1')
            axes[1, 1].set_ylabel('Valor Absoluto del Loading')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            save_path = os.path.join(self.results_dir, f"{save_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Análisis PCA guardado en: {save_path}")
        
        show_plot_briefly(1.0)
    
    def plot_segmentation_comparison(self, original_image: np.ndarray, 
                                   true_mask: np.ndarray,
                                   predictions: Dict[str, np.ndarray],
                                   image_id: str = "sample",
                                   save_name: Optional[str] = None):
        """
        Compara segmentaciones de diferentes métodos
        
        Args:
            original_image: Imagen original RGB
            true_mask: Máscara verdadera
            predictions: Diccionario {method_name: predicted_mask}
            image_id: Identificador de la imagen
            save_name: Nombre del archivo para guardar
        """
        n_methods = len(predictions)
        fig, axes = plt.subplots(2, n_methods + 1, figsize=(4*(n_methods+1), 8))
        
        # Fila superior: imagen original y máscara verdadera
        axes[0, 0].imshow(original_image.astype(np.uint8))
        axes[0, 0].set_title('Imagen Original')
        axes[0, 0].axis('off')
        
        axes[1, 0].imshow(true_mask, cmap='gray')
        axes[1, 0].set_title('Máscara de Referencia')
        axes[1, 0].axis('off')
        
        # Predicciones de cada método
        for i, (method_name, prediction) in enumerate(predictions.items()):
            col_idx = i + 1
            
            # Imagen original con contorno de predicción
            axes[0, col_idx].imshow(original_image.astype(np.uint8))
            
            # Agregar contorno de la predicción
            contours = self._find_contours(prediction)
            for contour in contours:
                axes[0, col_idx].plot(contour[:, 1], contour[:, 0], 'r-', linewidth=2)
            
            axes[0, col_idx].set_title(f'{method_name}\n(con contorno)')
            axes[0, col_idx].axis('off')
            
            # Máscara predicha
            axes[1, col_idx].imshow(prediction, cmap='gray')
            axes[1, col_idx].set_title(f'{method_name}\n(máscara)')
            axes[1, col_idx].axis('off')
        
        plt.suptitle(f'Comparación de Segmentación - {image_id}', fontsize=16)
        plt.tight_layout()
        
        if save_name:
            save_path = os.path.join(self.results_dir, f"{save_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparación de segmentación guardada en: {save_path}")
        
        show_plot_briefly(1.0)
    
    def _find_contours(self, binary_mask: np.ndarray) -> List[np.ndarray]:
        """
        Encuentra contornos de una máscara binaria
        
        Args:
            binary_mask: Máscara binaria
            
        Returns:
            Lista de contornos
        """
        try:
            import cv2
            contours, _ = cv2.findContours(binary_mask.astype(np.uint8), 
                                         cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return [contour.squeeze() for contour in contours if len(contour) > 3]
        except ImportError:
            # Fallback simple si opencv no está disponible
            return []
    
    def plot_metrics_comparison(self, results_comparison: Dict, 
                               save_name: Optional[str] = None):
        """
        Visualiza comparación de métricas entre métodos
        
        Args:
            results_comparison: Resultado de compare_methods()
            save_name: Nombre del archivo para guardar
        """
        methods = results_comparison['methods']
        metrics_to_plot = ['accuracy', 'precision', 'sensitivity', 'specificity', 
                          'f1_score', 'mean_jaccard']
        
        # Filtrar métricas disponibles
        available_metrics = [m for m in metrics_to_plot if m in results_comparison]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        colors = [self.colors['bayesian_rgb'], self.colors['bayesian_pca'], 
                 self.colors['kmeans']][:len(methods)]
        
        for i, metric in enumerate(available_metrics):
            if i >= len(axes):
                break
                
            values = results_comparison[metric]['values']
            
            bars = axes[i].bar(methods, values, color=colors)
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_ylabel('Valor')
            axes[i].set_ylim([0, 1])
            axes[i].grid(True, alpha=0.3)
            
            # Agregar valores sobre las barras
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
            
            # Rotar etiquetas si es necesario
            if len(max(methods, key=len)) > 10:
                axes[i].tick_params(axis='x', rotation=45)
        
        # Ocultar subplots no utilizados
        for i in range(len(available_metrics), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Comparación de Métricas entre Métodos', fontsize=16)
        plt.tight_layout()
        
        if save_name:
            save_path = os.path.join(self.results_dir, f"{save_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparación de métricas guardada en: {save_path}")
        
        show_plot_briefly(1.0)
    
    def plot_jaccard_distribution(self, image_metrics: Dict, 
                                 save_name: Optional[str] = None):
        """
        Visualiza distribución de índices de Jaccard por método
        
        Args:
            image_metrics: Métricas a nivel de imagen por método
            save_name: Nombre del archivo para guardar
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Histogramas de distribución
        for method_name, metrics in image_metrics.items():
            if 'individual_jaccard' in metrics:
                jaccard_scores = metrics['individual_jaccard']
                axes[0].hist(jaccard_scores, bins=20, alpha=0.7, 
                           label=f'{method_name} (μ={np.mean(jaccard_scores):.3f})')
        
        axes[0].set_title('Distribución de Índices de Jaccard')
        axes[0].set_xlabel('Índice de Jaccard')
        axes[0].set_ylabel('Frecuencia')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Box plots
        method_names = []
        jaccard_data = []
        
        for method_name, metrics in image_metrics.items():
            if 'individual_jaccard' in metrics:
                method_names.append(method_name)
                jaccard_data.append(metrics['individual_jaccard'])
        
        if jaccard_data:
            box_plot = axes[1].boxplot(jaccard_data, labels=method_names, patch_artist=True)
            
            # Colorear boxes
            colors = [self.colors['bayesian_rgb'], self.colors['bayesian_pca'], 
                     self.colors['kmeans']]
            for patch, color in zip(box_plot['boxes'], colors[:len(box_plot['boxes'])]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        axes[1].set_title('Distribución de Índices de Jaccard por Método')
        axes[1].set_ylabel('Índice de Jaccard')
        axes[1].grid(True, alpha=0.3)
        
        if len(max(method_names, key=len)) > 10:
            axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_name:
            save_path = os.path.join(self.results_dir, f"{save_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Distribución de Jaccard guardada en: {save_path}")
        
        show_plot_briefly(1.0)
    
    def create_results_summary_table(self, results_comparison: Dict,
                                   save_name: Optional[str] = None) -> pd.DataFrame:
        """
        Crea tabla resumen de resultados
        
        Args:
            results_comparison: Resultado de compare_methods()
            save_name: Nombre del archivo para guardar
            
        Returns:
            DataFrame con resultados
        """
        methods = results_comparison['methods']
        
        # Métricas principales
        main_metrics = ['accuracy', 'precision', 'sensitivity', 'specificity', 
                       'f1_score', 'mean_jaccard']
        
        # Crear DataFrame
        summary_data = {}
        for metric in main_metrics:
            if metric in results_comparison:
                summary_data[metric.replace('_', ' ').title()] = results_comparison[metric]['values']
        
        df = pd.DataFrame(summary_data, index=methods)
        df = df.round(4)
        
        # Guardar como CSV
        if save_name:
            csv_path = os.path.join(self.results_dir, f"{save_name}.csv")
            df.to_csv(csv_path)
            print(f"Tabla resumen guardada en: {csv_path}")
        
        return df
    
    def generate_final_report_figures(self, all_data: Dict):
        """
        Genera todas las figuras necesarias para el informe final
        
        Args:
            all_data: Diccionario con todos los datos y resultados del proyecto
        """
        print("Generando figuras para el informe final...")
        
        # 1. Histogramas RGB
        if 'processed_data' in all_data:
            train_data = all_data['processed_data']['train']
            self.plot_rgb_histograms(
                train_data['lesion_pixels'], 
                train_data['non_lesion_pixels'],
                save_name="fig_histogramas_rgb"
            )
        
        # 2. Estadísticas del dataset
        if 'dataset_stats' in all_data:
            self.plot_dataset_statistics(
                all_data['dataset_stats'],
                save_name="fig_estadisticas_dataset"
            )
        
        # 3. Análisis PCA
        if 'pca_info' in all_data:
            self.plot_pca_analysis(
                all_data['pca_info'],
                save_name="fig_analisis_pca"
            )
        
        # 4. Comparación de métricas
        if 'metrics_comparison' in all_data:
            self.plot_metrics_comparison(
                all_data['metrics_comparison'],
                save_name="fig_comparacion_metricas"
            )
        
        # 5. Distribución de Jaccard
        if 'image_metrics' in all_data:
            self.plot_jaccard_distribution(
                all_data['image_metrics'],
                save_name="fig_distribucion_jaccard"
            )
        
        # 6. Tabla resumen
        if 'metrics_comparison' in all_data:
            self.create_results_summary_table(
                all_data['metrics_comparison'],
                save_name="tabla_resumen_resultados"
            )
        
        print(f"Todas las figuras han sido guardadas en: {self.results_dir}")


if __name__ == "__main__":
    # Ejemplo de uso
    viz = VisualizationManager("results")
    
    # Datos sintéticos para prueba
    np.random.seed(42)
    
    # Simular píxeles RGB
    lesion_pixels = np.random.normal([150, 70, 60], [20, 15, 15], (1000, 3))
    non_lesion_pixels = np.random.normal([80, 120, 140], [25, 20, 20], (3000, 3))
    
    # Visualizar histogramas
    viz.plot_rgb_histograms(lesion_pixels, non_lesion_pixels, save_name="test_histograms")
    
    print("Módulo de visualización probado exitosamente!")
