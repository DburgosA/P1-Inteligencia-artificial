"""
Implementación de clasificación no supervisada usando K-Means
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.cluster import KMeans
import cv2
import matplotlib.pyplot as plt
import time

def show_plot_briefly(duration=1.0):
    """Muestra el plot actual por un tiempo específico y luego lo cierra"""
    plt.show(block=False)
    plt.pause(duration)
    plt.close()


class KMeansClassifier:
    """
    Clasificador no supervisado basado en K-Means para segmentación de lesiones
    """
    
    def __init__(self, n_clusters: int = 2, seed: int = 42, max_iter: int = 300):
        """
        Inicializa el clasificador K-Means
        
        Args:
            n_clusters: Número de clusters (por defecto 2 para lesión vs no-lesión)
            seed: Semilla para reproducibilidad
            max_iter: Número máximo de iteraciones
        """
        self.n_clusters = n_clusters
        self.seed = seed
        self.max_iter = max_iter
        
        # Modelo K-Means
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=seed,
            max_iter=max_iter,
            n_init=10
        )
        
        # Estado del modelo
        self.is_fitted = False
        self.cluster_centers_ = None
        self.cluster_to_class_mapping = None
    
    def fit_predict_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aplica K-Means a una imagen individual
        
        Args:
            image: Imagen RGB (H, W, 3)
            
        Returns:
            Tupla con máscara de clusters (H, W) y centroides (n_clusters, 3)
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("La imagen debe ser RGB (H, W, 3)")
        
        # Reshape imagen a vector de píxeles
        h, w, c = image.shape
        pixels = image.reshape(-1, c)
        
        # Aplicar K-Means
        cluster_labels = self.kmeans.fit_predict(pixels)
        
        # Obtener centroides
        centroids = self.kmeans.cluster_centers_
        
        # Reshape de vuelta a imagen
        cluster_mask = cluster_labels.reshape(h, w)
        
        self.is_fitted = True
        self.cluster_centers_ = centroids
        
        return cluster_mask, centroids
    
    def assign_clusters_to_classes(self, cluster_mask: np.ndarray, 
                                  reference_mask: np.ndarray) -> Dict[int, int]:
        """
        Asigna clusters a clases basado en superposición con máscara de referencia
        
        Args:
            cluster_mask: Máscara de clusters (H, W)
            reference_mask: Máscara de referencia (H, W) con 1=lesión, 0=no-lesión
            
        Returns:
            Diccionario mapeando cluster_id -> class_label (0 o 1)
        """
        if cluster_mask.shape != reference_mask.shape:
            raise ValueError("Las máscaras deben tener el mismo tamaño")
        
        unique_clusters = np.unique(cluster_mask)
        cluster_to_class = {}
        
        for cluster_id in unique_clusters:
            # Píxeles que pertenecen a este cluster
            cluster_pixels = (cluster_mask == cluster_id)
            
            # Calcular superposición con cada clase
            overlap_lesion = np.sum(cluster_pixels & (reference_mask == 1))
            overlap_non_lesion = np.sum(cluster_pixels & (reference_mask == 0))
            
            # Asignar a la clase con mayor superposición
            if overlap_lesion > overlap_non_lesion:
                cluster_to_class[cluster_id] = 1  # Lesión
            else:
                cluster_to_class[cluster_id] = 0  # No-lesión
        
        return cluster_to_class
    
    def convert_clusters_to_binary(self, cluster_mask: np.ndarray, 
                                  cluster_mapping: Dict[int, int]) -> np.ndarray:
        """
        Convierte máscara de clusters a máscara binaria usando el mapeo
        
        Args:
            cluster_mask: Máscara de clusters (H, W)
            cluster_mapping: Mapeo cluster_id -> class_label
            
        Returns:
            Máscara binaria (H, W) con 1=lesión, 0=no-lesión
        """
        binary_mask = np.zeros_like(cluster_mask)
        
        for cluster_id, class_label in cluster_mapping.items():
            binary_mask[cluster_mask == cluster_id] = class_label
        
        return binary_mask
    
    def process_single_image(self, image: np.ndarray, 
                           reference_mask: np.ndarray) -> Tuple[np.ndarray, Dict[int, int], Dict]:
        """
        Procesa una imagen completa: clustering + asignación de clases
        
        Args:
            image: Imagen RGB (H, W, 3)
            reference_mask: Máscara de referencia (H, W)
            
        Returns:
            Tupla con (máscara_binaria, mapeo_clusters, info_adicional)
        """
        # Aplicar K-Means
        cluster_mask, centroids = self.fit_predict_image(image)
        
        # Asignar clusters a clases
        cluster_mapping = self.assign_clusters_to_classes(cluster_mask, reference_mask)
        
        # Convertir a máscara binaria
        binary_prediction = self.convert_clusters_to_binary(cluster_mask, cluster_mapping)
        
        # Información adicional
        info = {
            'centroids': centroids,
            'cluster_mask': cluster_mask,
            'n_iterations': self.kmeans.n_iter_,
            'inertia': self.kmeans.inertia_
        }
        
        return binary_prediction, cluster_mapping, info
    
    def process_image_list(self, images: List[np.ndarray], 
                          reference_masks: List[np.ndarray]) -> Dict:
        """
        Procesa múltiples imágenes
        
        Args:
            images: Lista de imágenes RGB
            reference_masks: Lista de máscaras de referencia correspondientes
            
        Returns:
            Diccionario con resultados de todas las imágenes
        """
        if len(images) != len(reference_masks):
            raise ValueError("Debe haber igual número de imágenes y máscaras")
        
        results = {
            'binary_predictions': [],
            'cluster_mappings': [],
            'cluster_masks': [],
            'centroids': [],
            'inertias': [],
            'n_iterations': []
        }
        
        print(f"Procesando {len(images)} imágenes con K-Means...")
        
        for i, (image, ref_mask) in enumerate(zip(images, reference_masks)):
            print(f"  Procesando imagen {i+1}/{len(images)}")
            
            # Procesar imagen
            binary_pred, cluster_mapping, info = self.process_single_image(image, ref_mask)
            
            # Almacenar resultados
            results['binary_predictions'].append(binary_pred)
            results['cluster_mappings'].append(cluster_mapping)
            results['cluster_masks'].append(info['cluster_mask'])
            results['centroids'].append(info['centroids'])
            results['inertias'].append(info['inertia'])
            results['n_iterations'].append(info['n_iterations'])
        
        # Estadísticas agregadas
        results['mean_inertia'] = np.mean(results['inertias'])
        results['mean_iterations'] = np.mean(results['n_iterations'])
        
        print(f"K-Means completado:")
        print(f"  Inercia promedio: {results['mean_inertia']:.2f}")
        print(f"  Iteraciones promedio: {results['mean_iterations']:.1f}")
        
        return results
    
    def extract_pixel_predictions(self, images: List[np.ndarray], 
                                 reference_masks: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extrae predicciones a nivel de píxel para evaluación
        
        Args:
            images: Lista de imágenes RGB
            reference_masks: Lista de máscaras de referencia
            
        Returns:
            Tupla con (predicciones_pixel, etiquetas_verdaderas)
        """
        results = self.process_image_list(images, reference_masks)
        
        # Concatenar todas las predicciones de píxeles
        all_predictions = []
        all_true_labels = []
        
        for binary_pred, ref_mask in zip(results['binary_predictions'], reference_masks):
            all_predictions.extend(binary_pred.flatten())
            all_true_labels.extend(ref_mask.flatten())
        
        return np.array(all_predictions), np.array(all_true_labels)
    
    def get_cluster_statistics(self, results: Dict) -> Dict:
        """
        Calcula estadísticas de los clusters
        
        Args:
            results: Resultados de process_image_list()
            
        Returns:
            Diccionario con estadísticas
        """
        stats = {
            'n_images': len(results['binary_predictions']),
            'cluster_consistency': [],
            'centroid_distances': []
        }
        
        # Analizar consistencia de mapeo de clusters
        for mapping in results['cluster_mappings']:
            # Verificar si el mapeo es consistente (ej. cluster 0 siempre es la misma clase)
            stats['cluster_consistency'].append(mapping)
        
        # Calcular distancias entre centroides
        for centroids in results['centroids']:
            if len(centroids) >= 2:
                dist = np.linalg.norm(centroids[0] - centroids[1])
                stats['centroid_distances'].append(dist)
        
        stats['mean_centroid_distance'] = np.mean(stats['centroid_distances']) if stats['centroid_distances'] else 0
        
        return stats
    
    def visualize_clustering_result(self, image: np.ndarray, cluster_mask: np.ndarray,
                                  reference_mask: np.ndarray, binary_prediction: np.ndarray,
                                  save_path: Optional[str] = None):
        """
        Visualiza resultado del clustering para una imagen
        
        Args:
            image: Imagen original RGB
            cluster_mask: Máscara de clusters
            reference_mask: Máscara de referencia
            binary_prediction: Predicción binaria final
            save_path: Ruta para guardar la visualización
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Imagen original
        axes[0, 0].imshow(image.astype(np.uint8))
        axes[0, 0].set_title('Imagen Original')
        axes[0, 0].axis('off')
        
        # Máscara de referencia
        axes[0, 1].imshow(reference_mask, cmap='gray')
        axes[0, 1].set_title('Máscara de Referencia')
        axes[0, 1].axis('off')
        
        # Clusters K-Means
        axes[1, 0].imshow(cluster_mask, cmap='viridis')
        axes[1, 0].set_title('Clusters K-Means')
        axes[1, 0].axis('off')
        
        # Predicción binaria final
        axes[1, 1].imshow(binary_prediction, cmap='gray')
        axes[1, 1].set_title('Predicción Final')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualización guardada en: {save_path}")
        
        show_plot_briefly(1.0)


def apply_kmeans_to_dataset(images: List[np.ndarray], 
                           masks: List[np.ndarray],
                           n_clusters: int = 2,
                           seed: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Función de conveniencia para aplicar K-Means a un conjunto de datos
    
    Args:
        images: Lista de imágenes RGB
        masks: Lista de máscaras de referencia
        n_clusters: Número de clusters
        seed: Semilla para reproducibilidad
        
    Returns:
        Tupla con (predicciones_pixel, etiquetas_verdaderas, resultados_detallados)
    """
    classifier = KMeansClassifier(n_clusters=n_clusters, seed=seed)
    
    # Procesar todas las imágenes
    detailed_results = classifier.process_image_list(images, masks)
    
    # Extraer predicciones a nivel de píxel
    pixel_predictions, true_labels = classifier.extract_pixel_predictions(images, masks)
    
    return pixel_predictions, true_labels, detailed_results


if __name__ == "__main__":
    # Ejemplo de uso con datos sintéticos
    print("Probando K-Means con datos sintéticos...")
    
    # Crear imagen sintética
    np.random.seed(42)
    height, width = 100, 100
    
    # Imagen con dos regiones distintas
    image = np.zeros((height, width, 3))
    
    # Región 1 (fondo): colores azulados
    image[:50, :, 0] = np.random.normal(50, 15, (50, width))   # R bajo
    image[:50, :, 1] = np.random.normal(80, 15, (50, width))   # G medio
    image[:50, :, 2] = np.random.normal(120, 15, (50, width))  # B alto
    
    # Región 2 (lesión): colores rojizos
    image[50:, :, 0] = np.random.normal(150, 20, (50, width))  # R alto
    image[50:, :, 1] = np.random.normal(70, 20, (50, width))   # G bajo
    image[50:, :, 2] = np.random.normal(60, 20, (50, width))   # B bajo
    
    # Asegurar rango válido [0, 255]
    image = np.clip(image, 0, 255)
    
    # Máscara de referencia
    reference_mask = np.zeros((height, width))
    reference_mask[50:, :] = 1  # Parte inferior es lesión
    
    # Aplicar K-Means
    classifier = KMeansClassifier(n_clusters=2, seed=42)
    binary_prediction, cluster_mapping, info = classifier.process_single_image(image, reference_mask)
    
    print(f"Mapeo de clusters: {cluster_mapping}")
    print(f"Centroides: {info['centroids']}")
    print(f"Inercia: {info['inertia']:.2f}")
    print(f"Iteraciones: {info['n_iterations']}")
    
    # Visualizar resultado
    classifier.visualize_clustering_result(
        image, info['cluster_mask'], reference_mask, binary_prediction
    )