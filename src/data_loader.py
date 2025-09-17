"""
Módulo para carga y manejo del dataset ISIC
"""

import os
import glob
import numpy as np
import cv2
from PIL import Image
import random
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
import time

def show_plot_briefly(duration=1.0):
    """Muestra el plot actual por un tiempo específico y luego lo cierra"""
    plt.show(block=False)
    plt.pause(duration)
    plt.close()

class ISICDataLoader:
    """
    Clase para cargar y manejar el dataset ISIC
    """
    
    def __init__(self, dataset_path: str, seed: int = 42):
        """
        Inicializa el cargador de datos
        
        Args:
            dataset_path: Ruta al directorio del dataset
            seed: Semilla para reproducibilidad
        """
        self.dataset_path = dataset_path
        self.seed = seed
        self.set_seed()
        
        # Encontrar todos los archivos
        self.image_files = self._find_image_files()
        self.mask_files = self._find_mask_files()
        
        # Verificar correspondencia
        self._verify_correspondence()
        
        print(f"Dataset cargado: {len(self.image_files)} imágenes encontradas")
    
    def set_seed(self):
        """Establece semillas para reproducibilidad"""
        random.seed(self.seed)
        np.random.seed(self.seed)
    
    def _find_image_files(self) -> List[str]:
        """Encuentra todos los archivos de imágenes .jpg"""
        pattern = os.path.join(self.dataset_path, "*.jpg")
        files = glob.glob(pattern)
        return sorted(files)
    
    def _find_mask_files(self) -> List[str]:
        """Encuentra todos los archivos de máscaras _expert.png"""
        pattern = os.path.join(self.dataset_path, "*_expert.png")
        files = glob.glob(pattern)
        return sorted(files)
    
    def _verify_correspondence(self):
        """Verifica que cada imagen tenga su máscara correspondiente"""
        image_ids = set()
        mask_ids = set()
        
        for img_file in self.image_files:
            basename = os.path.basename(img_file)
            image_id = basename.replace('.jpg', '')
            image_ids.add(image_id)
        
        for mask_file in self.mask_files:
            basename = os.path.basename(mask_file)
            mask_id = basename.replace('_expert.png', '')
            mask_ids.add(mask_id)
        
        if image_ids != mask_ids:
            missing_masks = image_ids - mask_ids
            missing_images = mask_ids - image_ids
            
            if missing_masks:
                print(f"Advertencia: Máscaras faltantes para: {missing_masks}")
            if missing_images:
                print(f"Advertencia: Imágenes faltantes para: {missing_images}")
            
            # Mantener solo los que tienen correspondencia
            common_ids = image_ids & mask_ids
            self.image_files = [f for f in self.image_files 
                              if os.path.basename(f).replace('.jpg', '') in common_ids]
            self.mask_files = [f for f in self.mask_files 
                             if os.path.basename(f).replace('_expert.png', '') in common_ids]
        
        print(f"Verificación completada: {len(self.image_files)} pares imagen-máscara válidos")
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Carga una imagen RGB
        
        Args:
            image_path: Ruta a la imagen
            
        Returns:
            Array numpy con la imagen RGB (H, W, 3)
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")
        
        # Convertir de BGR a RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image.astype(np.float32)
    
    def load_mask(self, mask_path: str) -> np.ndarray:
        """
        Carga una máscara binaria
        
        Args:
            mask_path: Ruta a la máscara
            
        Returns:
            Array numpy con la máscara binaria (H, W)
        """
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"No se pudo cargar la máscara: {mask_path}")
        
        # Convertir a binario (0 o 1)
        mask = (mask > 127).astype(np.uint8)
        return mask
    
    def split_dataset(self, train_ratio: float = 0.6, val_ratio: float = 0.2, 
                     test_ratio: float = 0.2) -> Dict[str, List[Tuple[str, str]]]:
        """
        Divide el dataset en entrenamiento, validación y test
        
        Args:
            train_ratio: Proporción para entrenamiento
            val_ratio: Proporción para validación  
            test_ratio: Proporción para test
            
        Returns:
            Diccionario con las divisiones
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Las proporciones deben sumar 1.0")
        
        # Crear pares imagen-máscara
        pairs = list(zip(self.image_files, self.mask_files))
        
        # Mezclar aleatoriamente
        random.shuffle(pairs)
        
        total = len(pairs)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        
        train_pairs = pairs[:train_size]
        val_pairs = pairs[train_size:train_size + val_size]
        test_pairs = pairs[train_size + val_size:]
        
        split_data = {
            'train': train_pairs,
            'validation': val_pairs,
            'test': test_pairs
        }
        
        print(f"División del dataset:")
        print(f"  Entrenamiento: {len(train_pairs)} imágenes ({len(train_pairs)/total*100:.1f}%)")
        print(f"  Validación: {len(val_pairs)} imágenes ({len(val_pairs)/total*100:.1f}%)")
        print(f"  Test: {len(test_pairs)} imágenes ({len(test_pairs)/total*100:.1f}%)")
        
        return split_data
    
    def extract_pixels(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extrae píxeles de lesión y no-lesión basado en la máscara
        
        Args:
            image: Imagen RGB (H, W, 3)
            mask: Máscara binaria (H, W)
            
        Returns:
            Tupla con píxeles de lesión y no-lesión
        """
        lesion_pixels = image[mask == 1]      # Píxeles donde mask == 1
        non_lesion_pixels = image[mask == 0]  # Píxeles donde mask == 0
        
        return lesion_pixels, non_lesion_pixels
    
    def load_split_data(self, split_data: Dict, balanced_sampling: bool = True) -> Dict:
        """
        Carga datos divididos y extrae características
        
        Args:
            split_data: Resultado de split_dataset()
            balanced_sampling: Si aplicar muestreo equilibrado en entrenamiento
            
        Returns:
            Diccionario con datos procesados
        """
        processed_data = {}
        
        for split_name, pairs in split_data.items():
            print(f"\nProcesando {split_name}...")
            
            all_lesion_pixels = []
            all_non_lesion_pixels = []
            all_images = []
            all_masks = []
            
            for img_path, mask_path in pairs:
                # Cargar imagen y máscara
                image = self.load_image(img_path)
                mask = self.load_mask(mask_path)
                
                # Extraer píxeles
                lesion_pixels, non_lesion_pixels = self.extract_pixels(image, mask)
                
                all_lesion_pixels.append(lesion_pixels)
                all_non_lesion_pixels.append(non_lesion_pixels)
                all_images.append(image)
                all_masks.append(mask)
            
            # Concatenar todos los píxeles
            lesion_data = np.vstack(all_lesion_pixels) if all_lesion_pixels else np.array([]).reshape(0, 3)
            non_lesion_data = np.vstack(all_non_lesion_pixels) if all_non_lesion_pixels else np.array([]).reshape(0, 3)
            
            # Aplicar muestreo equilibrado solo en entrenamiento
            if split_name == 'train' and balanced_sampling and len(lesion_data) > 0 and len(non_lesion_data) > 0:
                lesion_data, non_lesion_data = self._balanced_sampling(lesion_data, non_lesion_data)
            
            processed_data[split_name] = {
                'lesion_pixels': lesion_data,
                'non_lesion_pixels': non_lesion_data,
                'images': all_images,
                'masks': all_masks,
                'image_paths': [pair[0] for pair in pairs],
                'mask_paths': [pair[1] for pair in pairs]
            }
            
            print(f"  Píxeles de lesión: {len(lesion_data):,}")
            print(f"  Píxeles de no-lesión: {len(non_lesion_data):,}")
        
        return processed_data
    
    def _balanced_sampling(self, lesion_pixels: np.ndarray, 
                          non_lesion_pixels: np.ndarray, 
                          ratio: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aplica muestreo equilibrado entre clases
        
        Args:
            lesion_pixels: Píxeles de lesión
            non_lesion_pixels: Píxeles de no-lesión
            ratio: Ratio lesión:no-lesión (1.0 = equilibrado)
            
        Returns:
            Píxeles balanceados
        """
        n_lesion = len(lesion_pixels)
        n_non_lesion = len(non_lesion_pixels)
        
        if n_lesion == 0 or n_non_lesion == 0:
            return lesion_pixels, non_lesion_pixels
        
        # Calcular tamaño objetivo
        target_non_lesion = int(n_lesion * ratio)
        
        if target_non_lesion > n_non_lesion:
            # Submuestrear lesión
            target_lesion = int(n_non_lesion / ratio)
            indices_lesion = np.random.choice(n_lesion, target_lesion, replace=False)
            lesion_balanced = lesion_pixels[indices_lesion]
            non_lesion_balanced = non_lesion_pixels
        else:
            # Submuestrear no-lesión
            indices_non_lesion = np.random.choice(n_non_lesion, target_non_lesion, replace=False)
            lesion_balanced = lesion_pixels
            non_lesion_balanced = non_lesion_pixels[indices_non_lesion]
        
        print(f"    Muestreo equilibrado aplicado:")
        print(f"      Lesión: {len(lesion_pixels):,} → {len(lesion_balanced):,}")
        print(f"      No-lesión: {len(non_lesion_pixels):,} → {len(non_lesion_balanced):,}")
        
        return lesion_balanced, non_lesion_balanced
    
    def get_dataset_statistics(self, processed_data: Dict) -> Dict:
        """
        Calcula estadísticas del dataset
        
        Args:
            processed_data: Datos procesados
            
        Returns:
            Diccionario con estadísticas
        """
        stats = {}
        
        for split_name, data in processed_data.items():
            lesion_pixels = data['lesion_pixels']
            non_lesion_pixels = data['non_lesion_pixels']
            
            stats[split_name] = {
                'n_images': len(data['images']),
                'n_lesion_pixels': len(lesion_pixels),
                'n_non_lesion_pixels': len(non_lesion_pixels),
                'total_pixels': len(lesion_pixels) + len(non_lesion_pixels)
            }
            
            # Estadísticas RGB por clase
            if len(lesion_pixels) > 0:
                stats[split_name]['lesion_rgb_mean'] = np.mean(lesion_pixels, axis=0)
                stats[split_name]['lesion_rgb_std'] = np.std(lesion_pixels, axis=0)
            
            if len(non_lesion_pixels) > 0:
                stats[split_name]['non_lesion_rgb_mean'] = np.mean(non_lesion_pixels, axis=0)
                stats[split_name]['non_lesion_rgb_std'] = np.std(non_lesion_pixels, axis=0)
        
        return stats
    
    def plot_rgb_histograms(self, processed_data: Dict, split_name: str = 'train', 
                           save_path: str = None):
        """
        Genera histogramas RGB por clase
        
        Args:
            processed_data: Datos procesados
            split_name: División a graficar
            save_path: Ruta para guardar la figura
        """
        if split_name not in processed_data:
            raise ValueError(f"Split '{split_name}' no encontrado")
        
        data = processed_data[split_name]
        lesion_pixels = data['lesion_pixels']
        non_lesion_pixels = data['non_lesion_pixels']
        
        if len(lesion_pixels) == 0 or len(non_lesion_pixels) == 0:
            print("No hay suficientes datos para generar histogramas")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        channels = ['R', 'G', 'B']
        colors = ['red', 'green', 'blue']
        
        for i, (channel, color) in enumerate(zip(channels, colors)):
            # Histograma de lesión
            axes[0, i].hist(lesion_pixels[:, i], bins=50, alpha=0.7, 
                           color=color, label='Lesión', density=True)
            axes[0, i].set_title(f'Canal {channel} - Lesión')
            axes[0, i].set_xlabel('Intensidad')
            axes[0, i].set_ylabel('Densidad')
            axes[0, i].grid(True, alpha=0.3)
            
            # Histograma de no-lesión
            axes[1, i].hist(non_lesion_pixels[:, i], bins=50, alpha=0.7, 
                           color=color, label='No-lesión', density=True)
            axes[1, i].set_title(f'Canal {channel} - No-lesión')
            axes[1, i].set_xlabel('Intensidad')
            axes[1, i].set_ylabel('Densidad')
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Histogramas guardados en: {save_path}")
        
        show_plot_briefly(1.0)


# Funciones auxiliares
def create_combined_features_and_labels(lesion_pixels: np.ndarray, 
                                       non_lesion_pixels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combina píxeles de ambas clases y crea etiquetas
    
    Args:
        lesion_pixels: Píxeles de lesión (N1, 3)
        non_lesion_pixels: Píxeles de no-lesión (N2, 3)
        
    Returns:
        Tupla con características (N1+N2, 3) y etiquetas (N1+N2,)
    """
    if len(lesion_pixels) == 0 and len(non_lesion_pixels) == 0:
        return np.array([]).reshape(0, 3), np.array([])
    
    # Combinar características
    features = np.vstack([lesion_pixels, non_lesion_pixels])
    
    # Crear etiquetas (1 para lesión, 0 para no-lesión)
    labels = np.hstack([
        np.ones(len(lesion_pixels)),
        np.zeros(len(non_lesion_pixels))
    ])
    
    return features, labels


if __name__ == "__main__":
    # Ejemplo de uso
    dataset_path = "dataset"
    
    # Crear cargador de datos
    loader = ISICDataLoader(dataset_path, seed=42)
    
    # Dividir dataset
    split_data = loader.split_dataset(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    
    # Cargar y procesar datos
    processed_data = loader.load_split_data(split_data, balanced_sampling=True)
    
    # Obtener estadísticas
    stats = loader.get_dataset_statistics(processed_data)
    
    # Mostrar estadísticas
    for split, stat in stats.items():
        print(f"\n{split.upper()}:")
        print(f"  Imágenes: {stat['n_images']}")
        print(f"  Píxeles totales: {stat['total_pixels']:,}")
        if 'lesion_rgb_mean' in stat:
            print(f"  RGB medio lesión: {stat['lesion_rgb_mean']}")
        if 'non_lesion_rgb_mean' in stat:
            print(f"  RGB medio no-lesión: {stat['non_lesion_rgb_mean']}")
    
    # Generar histogramas
    loader.plot_rgb_histograms(processed_data, 'train', 'results/rgb_histograms.png')