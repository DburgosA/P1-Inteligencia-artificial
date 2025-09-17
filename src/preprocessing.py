"""
Módulo de preprocesamiento para datos de segmentación de lesiones
"""

import numpy as np
from typing import Tuple, Dict, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings


class DataPreprocessor:
    """
    Clase para preprocesamiento de datos con prevención de data leakage
    """
    
    def __init__(self, seed: int = 42):
        """
        Inicializa el preprocesador
        
        Args:
            seed: Semilla para reproducibilidad
        """
        self.seed = seed
        np.random.seed(seed)
        
        # Almacenar transformadores entrenados solo en datos de entrenamiento
        self.scaler = None
        self.pca = None
        self.is_fitted = False
    
    def fit_normalizer(self, train_lesion_pixels: np.ndarray, 
                      train_non_lesion_pixels: np.ndarray) -> 'DataPreprocessor':
        """
        Ajusta el normalizador usando solo datos de entrenamiento
        
        Args:
            train_lesion_pixels: Píxeles de lesión de entrenamiento
            train_non_lesion_pixels: Píxeles de no-lesión de entrenamiento
            
        Returns:
            Self para method chaining
        """
        if len(train_lesion_pixels) == 0 and len(train_non_lesion_pixels) == 0:
            raise ValueError("No hay datos de entrenamiento para ajustar el normalizador")
        
        # Combinar todos los píxeles de entrenamiento
        all_train_pixels = np.vstack([train_lesion_pixels, train_non_lesion_pixels])
        
        # Inicializar y ajustar scaler
        self.scaler = StandardScaler()
        self.scaler.fit(all_train_pixels)
        
        print("Normalizador ajustado con estadísticas de entrenamiento:")
        print(f"  Medias por canal: {self.scaler.mean_}")
        print(f"  Desviaciones por canal: {self.scaler.scale_}")
        
        return self
    
    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        """
        Normaliza datos usando estadísticas de entrenamiento
        
        Args:
            data: Datos a normalizar (N, 3)
            
        Returns:
            Datos normalizados
        """
        if self.scaler is None:
            raise ValueError("Debe ajustar el normalizador primero con fit_normalizer()")
        
        if len(data) == 0:
            return data
        
        return self.scaler.transform(data)
    
    def fit_pca(self, train_lesion_pixels: np.ndarray, 
               train_non_lesion_pixels: np.ndarray,
               variance_threshold: float = 0.95) -> 'DataPreprocessor':
        """
        Ajusta PCA usando solo datos de entrenamiento normalizados
        
        Args:
            train_lesion_pixels: Píxeles de lesión de entrenamiento (normalizados)
            train_non_lesion_pixels: Píxeles de no-lesión de entrenamiento (normalizados)
            variance_threshold: Porcentaje de varianza a preservar
            
        Returns:
            Self para method chaining
        """
        if len(train_lesion_pixels) == 0 and len(train_non_lesion_pixels) == 0:
            raise ValueError("No hay datos de entrenamiento para ajustar PCA")
        
        # Combinar datos de entrenamiento
        all_train_pixels = np.vstack([train_lesion_pixels, train_non_lesion_pixels])
        
        # Ajustar PCA inicial para determinar número de componentes
        pca_temp = PCA()
        pca_temp.fit(all_train_pixels)
        
        # Determinar número de componentes necesarias
        cumsum_variance = np.cumsum(pca_temp.explained_variance_ratio_)
        n_components = np.argmax(cumsum_variance >= variance_threshold) + 1
        
        # Asegurar que tenemos al menos 1 componente y no más que las dimensiones originales
        n_components = max(1, min(n_components, all_train_pixels.shape[1]))
        
        # Ajustar PCA final con número óptimo de componentes
        self.pca = PCA(n_components=n_components, random_state=self.seed)
        self.pca.fit(all_train_pixels)
        
        print(f"PCA ajustado:")
        print(f"  Componentes seleccionadas: {n_components}")
        print(f"  Varianza explicada por componente: {self.pca.explained_variance_ratio_}")
        print(f"  Varianza total explicada: {np.sum(self.pca.explained_variance_ratio_):.4f}")
        
        return self
    
    def apply_pca(self, data: np.ndarray) -> np.ndarray:
        """
        Aplica transformación PCA a los datos
        
        Args:
            data: Datos normalizados a transformar
            
        Returns:
            Datos transformados por PCA
        """
        if self.pca is None:
            raise ValueError("Debe ajustar PCA primero con fit_pca()")
        
        if len(data) == 0:
            return np.array([]).reshape(0, self.pca.n_components_)
        
        return self.pca.transform(data)
    
    def fit_and_transform_train(self, train_lesion_pixels: np.ndarray,
                               train_non_lesion_pixels: np.ndarray,
                               apply_pca: bool = False,
                               variance_threshold: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ajusta transformadores y aplica a datos de entrenamiento
        
        Args:
            train_lesion_pixels: Píxeles de lesión de entrenamiento
            train_non_lesion_pixels: Píxeles de no-lesión de entrenamiento
            apply_pca: Si aplicar PCA
            variance_threshold: Umbral de varianza para PCA
            
        Returns:
            Tupla con píxeles de lesión y no-lesión transformados
        """
        # Ajustar y aplicar normalización
        self.fit_normalizer(train_lesion_pixels, train_non_lesion_pixels)
        
        lesion_normalized = self.normalize_data(train_lesion_pixels)
        non_lesion_normalized = self.normalize_data(train_non_lesion_pixels)
        
        if apply_pca:
            # Ajustar y aplicar PCA
            self.fit_pca(lesion_normalized, non_lesion_normalized, variance_threshold)
            
            lesion_transformed = self.apply_pca(lesion_normalized)
            non_lesion_transformed = self.apply_pca(non_lesion_normalized)
            
            self.is_fitted = True
            return lesion_transformed, non_lesion_transformed
        else:
            self.is_fitted = True
            return lesion_normalized, non_lesion_normalized
    
    def transform(self, lesion_pixels: np.ndarray, 
                 non_lesion_pixels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aplica transformaciones ya ajustadas a nuevos datos
        
        Args:
            lesion_pixels: Píxeles de lesión a transformar
            non_lesion_pixels: Píxeles de no-lesión a transformar
            
        Returns:
            Tupla con píxeles transformados
        """
        if not self.is_fitted or self.scaler is None:
            raise ValueError("Debe ajustar las transformaciones primero con fit_and_transform_train()")
        
        # Aplicar normalización
        lesion_normalized = self.normalize_data(lesion_pixels)
        non_lesion_normalized = self.normalize_data(non_lesion_pixels)
        
        # Aplicar PCA si está disponible
        if self.pca is not None:
            lesion_transformed = self.apply_pca(lesion_normalized)
            non_lesion_transformed = self.apply_pca(non_lesion_normalized)
            return lesion_transformed, non_lesion_transformed
        else:
            return lesion_normalized, non_lesion_normalized
    
    def get_feature_dimension(self) -> int:
        """
        Obtiene la dimensión final de las características después de transformaciones
        
        Returns:
            Número de características
        """
        if self.pca is not None:
            return self.pca.n_components_
        elif self.scaler is not None:
            return len(self.scaler.mean_)
        else:
            return 3  # RGB original
    
    def get_pca_info(self) -> Optional[Dict]:
        """
        Obtiene información sobre la transformación PCA
        
        Returns:
            Diccionario con información de PCA o None si no se aplicó
        """
        if self.pca is None:
            return None
        
        return {
            'n_components': self.pca.n_components_,
            'explained_variance_ratio': self.pca.explained_variance_ratio_,
            'total_variance_explained': np.sum(self.pca.explained_variance_ratio_),
            'components': self.pca.components_
        }
    
    def inverse_transform_pca(self, data_pca: np.ndarray) -> np.ndarray:
        """
        Convierte datos de espacio PCA de vuelta a espacio normalizado
        
        Args:
            data_pca: Datos en espacio PCA
            
        Returns:
            Datos en espacio normalizado
        """
        if self.pca is None:
            raise ValueError("PCA no ha sido ajustado")
        
        if len(data_pca) == 0:
            return np.array([]).reshape(0, 3)
        
        return self.pca.inverse_transform(data_pca)
    
    def inverse_normalize(self, data_normalized: np.ndarray) -> np.ndarray:
        """
        Convierte datos normalizados de vuelta a escala original
        
        Args:
            data_normalized: Datos normalizados
            
        Returns:
            Datos en escala original
        """
        if self.scaler is None:
            raise ValueError("Scaler no ha sido ajustado")
        
        if len(data_normalized) == 0:
            return data_normalized
        
        return self.scaler.inverse_transform(data_normalized)


def process_dataset_splits(processed_data: Dict, apply_pca: bool = False, 
                          variance_threshold: float = 0.95, seed: int = 42) -> Tuple[Dict, DataPreprocessor]:
    """
    Procesa todas las divisiones del dataset con prevención de data leakage
    
    Args:
        processed_data: Datos divididos del DataLoader
        apply_pca: Si aplicar reducción de dimensionalidad
        variance_threshold: Umbral de varianza para PCA
        seed: Semilla para reproducibilidad
        
    Returns:
        Tupla con datos procesados y el preprocesador ajustado
    """
    preprocessor = DataPreprocessor(seed=seed)
    
    # Verificar que tengamos datos de entrenamiento
    if 'train' not in processed_data:
        raise ValueError("No se encontraron datos de entrenamiento")
    
    train_data = processed_data['train']
    
    # Ajustar transformaciones solo en entrenamiento
    train_lesion_transformed, train_non_lesion_transformed = preprocessor.fit_and_transform_train(
        train_data['lesion_pixels'],
        train_data['non_lesion_pixels'],
        apply_pca=apply_pca,
        variance_threshold=variance_threshold
    )
    
    # Aplicar transformaciones a todas las divisiones
    transformed_data = {}
    
    for split_name, data in processed_data.items():
        if split_name == 'train':
            # Usar datos ya transformados
            transformed_data[split_name] = {
                'lesion_pixels': train_lesion_transformed,
                'non_lesion_pixels': train_non_lesion_transformed,
                'images': data['images'],
                'masks': data['masks'],
                'image_paths': data['image_paths'],
                'mask_paths': data['mask_paths']
            }
        else:
            # Transformar usando parámetros del entrenamiento
            lesion_transformed, non_lesion_transformed = preprocessor.transform(
                data['lesion_pixels'],
                data['non_lesion_pixels']
            )
            
            transformed_data[split_name] = {
                'lesion_pixels': lesion_transformed,
                'non_lesion_pixels': non_lesion_transformed,
                'images': data['images'],
                'masks': data['masks'],
                'image_paths': data['image_paths'],
                'mask_paths': data['mask_paths']
            }
    
    # Imprimir información de transformación
    print(f"\nPreprocesamiento completado:")
    print(f"  Normalización aplicada: Sí")
    print(f"  PCA aplicado: {'Sí' if apply_pca else 'No'}")
    print(f"  Dimensión final: {preprocessor.get_feature_dimension()}")
    
    if apply_pca:
        pca_info = preprocessor.get_pca_info()
        print(f"  Varianza total explicada: {pca_info['total_variance_explained']:.4f}")
    
    return transformed_data, preprocessor


def validate_preprocessing(transformed_data: Dict, original_data: Dict):
    """
    Valida que el preprocesamiento se haya aplicado correctamente
    
    Args:
        transformed_data: Datos después del preprocesamiento
        original_data: Datos originales
    """
    print("\nValidación del preprocesamiento:")
    
    for split_name in transformed_data.keys():
        if split_name not in original_data:
            continue
        
        orig_lesion = original_data[split_name]['lesion_pixels']
        orig_non_lesion = original_data[split_name]['non_lesion_pixels']
        
        trans_lesion = transformed_data[split_name]['lesion_pixels']
        trans_non_lesion = transformed_data[split_name]['non_lesion_pixels']
        
        print(f"\n{split_name.upper()}:")
        
        # Verificar que los tamaños sean consistentes
        if len(orig_lesion) > 0 and len(trans_lesion) > 0:
            print(f"  Lesión - Original: {orig_lesion.shape}, Transformado: {trans_lesion.shape}")
            
            # Verificar medias (deben estar cerca de 0 después de normalización)
            if split_name == 'train':
                lesion_means = np.mean(trans_lesion, axis=0)
                print(f"  Medias de lesión (deben estar ~0): {lesion_means}")
        
        if len(orig_non_lesion) > 0 and len(trans_non_lesion) > 0:
            print(f"  No-lesión - Original: {orig_non_lesion.shape}, Transformado: {trans_non_lesion.shape}")
            
            # Verificar medias
            if split_name == 'train':
                non_lesion_means = np.mean(trans_non_lesion, axis=0)
                print(f"  Medias de no-lesión (deben estar ~0): {non_lesion_means}")


if __name__ == "__main__":
    # Ejemplo de uso
    from data_loader import ISICDataLoader, create_combined_features_and_labels
    
    # Cargar datos
    loader = ISICDataLoader("dataset", seed=42)
    split_data = loader.split_dataset()
    processed_data = loader.load_split_data(split_data)
    
    # Procesar sin PCA
    print("=== Procesamiento sin PCA ===")
    transformed_data_rgb, preprocessor_rgb = process_dataset_splits(
        processed_data, apply_pca=False, seed=42
    )
    validate_preprocessing(transformed_data_rgb, processed_data)
    
    # Procesar con PCA
    print("\n=== Procesamiento con PCA ===")
    transformed_data_pca, preprocessor_pca = process_dataset_splits(
        processed_data, apply_pca=True, variance_threshold=0.95, seed=42
    )
    validate_preprocessing(transformed_data_pca, processed_data)