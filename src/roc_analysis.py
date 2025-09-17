"""
Análisis ROC y selección de umbrales óptimos
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import warnings
import time

def show_plot_briefly(duration=1.0):
    """Muestra el plot actual por un tiempo específico y luego lo cierra"""
    plt.show(block=False)
    plt.pause(duration)
    plt.close()


class ROCAnalyzer:
    """
    Clase para análisis de curvas ROC y selección de umbrales óptimos
    """
    
    def __init__(self):
        self.roc_data = {}  # Almacenar datos ROC por método
    
    def calculate_roc_curve(self, y_true: np.ndarray, y_scores: np.ndarray, 
                           method_name: str = "Unknown") -> Dict:
        """
        Calcula curva ROC para un conjunto de datos
        
        Args:
            y_true: Etiquetas verdaderas (N,)
            y_scores: Scores de decisión (N,)
            method_name: Nombre del método
            
        Returns:
            Diccionario con datos de la curva ROC
        """
        # Calcular curva ROC
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        auc_score = auc(fpr, tpr)
        
        # Almacenar datos
        roc_data = {
            'method_name': method_name,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': auc_score,
            'y_true': y_true,
            'y_scores': y_scores
        }
        
        self.roc_data[method_name] = roc_data
        
        print(f"Curva ROC calculada para {method_name}:")
        print(f"  AUC: {auc_score:.4f}")
        print(f"  Puntos en la curva: {len(fpr)}")
        
        return roc_data
    
    def find_youden_threshold(self, y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float, Dict]:
        """
        Encuentra el umbral óptimo según el índice de Youden
        
        Args:
            y_true: Etiquetas verdaderas
            y_scores: Scores de decisión
            
        Returns:
            Tupla con (umbral_óptimo, índice_youden, info_adicional)
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        
        # Calcular índice de Youden (J = TPR - FPR)
        youden_index = tpr - fpr
        
        # Encontrar el punto óptimo
        optimal_idx = np.argmax(youden_index)
        optimal_threshold = thresholds[optimal_idx]
        optimal_youden = youden_index[optimal_idx]
        optimal_tpr = tpr[optimal_idx]
        optimal_fpr = fpr[optimal_idx]
        
        info = {
            'optimal_threshold': optimal_threshold,
            'youden_index': optimal_youden,
            'tpr': optimal_tpr,
            'fpr': optimal_fpr,
            'sensitivity': optimal_tpr,
            'specificity': 1 - optimal_fpr
        }
        
        return optimal_threshold, optimal_youden, info
    
    def find_eer_threshold(self, y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float, Dict]:
        """
        Encuentra el umbral en el Equal Error Rate (EER)
        
        Args:
            y_true: Etiquetas verdaderas
            y_scores: Scores de decisión
            
        Returns:
            Tupla con (umbral_eer, eer_value, info_adicional)
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        fnr = 1 - tpr  # False Negative Rate
        
        # Encontrar punto donde FPR ≈ FNR
        eer_diffs = np.abs(fpr - fnr)
        eer_idx = np.argmin(eer_diffs)
        
        eer_threshold = thresholds[eer_idx]
        eer_value = (fpr[eer_idx] + fnr[eer_idx]) / 2
        
        info = {
            'eer_threshold': eer_threshold,
            'eer_value': eer_value,
            'fpr': fpr[eer_idx],
            'fnr': fnr[eer_idx],
            'tpr': tpr[eer_idx],
            'sensitivity': tpr[eer_idx],
            'specificity': 1 - fpr[eer_idx]
        }
        
        return eer_threshold, eer_value, info
    
    def find_high_sensitivity_threshold(self, y_true: np.ndarray, y_scores: np.ndarray, 
                                       min_sensitivity: float = 0.9) -> Tuple[Optional[float], Dict]:
        """
        Encuentra umbral que garantiza sensibilidad mínima
        
        Args:
            y_true: Etiquetas verdaderas
            y_scores: Scores de decisión
            min_sensitivity: Sensibilidad mínima requerida
            
        Returns:
            Tupla con (umbral_óptimo, info_adicional)
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        
        # Encontrar puntos que cumplen la restricción de sensibilidad
        valid_indices = tpr >= min_sensitivity
        
        if not np.any(valid_indices):
            # No se puede alcanzar la sensibilidad requerida
            info = {
                'feasible': False,
                'max_achievable_sensitivity': np.max(tpr),
                'threshold': None,
                'sensitivity': None,
                'specificity': None,
                'fpr': None
            }
            return None, info
        
        # Entre los puntos válidos, seleccionar el que minimiza FPR (maximiza especificidad)
        valid_fpr = fpr[valid_indices]
        valid_tpr = tpr[valid_indices]
        valid_thresholds = thresholds[valid_indices]
        
        optimal_idx = np.argmin(valid_fpr)
        optimal_threshold = valid_thresholds[optimal_idx]
        optimal_sensitivity = valid_tpr[optimal_idx]
        optimal_specificity = 1 - valid_fpr[optimal_idx]
        
        info = {
            'feasible': True,
            'threshold': optimal_threshold,
            'sensitivity': optimal_sensitivity,
            'specificity': optimal_specificity,
            'fpr': valid_fpr[optimal_idx],
            'min_sensitivity_required': min_sensitivity
        }
        
        return optimal_threshold, info
    
    def comprehensive_threshold_analysis(self, y_true: np.ndarray, y_scores: np.ndarray,
                                       method_name: str = "Unknown",
                                       min_sensitivity: float = 0.9) -> Dict:
        """
        Análisis completo de selección de umbrales
        
        Args:
            y_true: Etiquetas verdaderas
            y_scores: Scores de decisión
            method_name: Nombre del método
            min_sensitivity: Sensibilidad mínima para restricción operativa
            
        Returns:
            Diccionario con todos los criterios de selección
        """
        # Calcular curva ROC
        roc_data = self.calculate_roc_curve(y_true, y_scores, method_name)
        
        # Criterio 1: Índice de Youden
        youden_threshold, youden_value, youden_info = self.find_youden_threshold(y_true, y_scores)
        
        # Criterio 2: Equal Error Rate
        eer_threshold, eer_value, eer_info = self.find_eer_threshold(y_true, y_scores)
        
        # Criterio 3: Alta sensibilidad
        high_sens_threshold, high_sens_info = self.find_high_sensitivity_threshold(
            y_true, y_scores, min_sensitivity
        )
        
        analysis = {
            'method_name': method_name,
            'auc': roc_data['auc'],
            'youden': {
                'threshold': youden_threshold,
                'index': youden_value,
                'sensitivity': youden_info['sensitivity'],
                'specificity': youden_info['specificity']
            },
            'eer': {
                'threshold': eer_threshold,
                'value': eer_value,
                'sensitivity': eer_info['sensitivity'],
                'specificity': eer_info['specificity']
            },
            'high_sensitivity': high_sens_info
        }
        
        return analysis
    
    def plot_roc_curve(self, method_name: str, save_path: Optional[str] = None):
        """
        Plotea curva ROC para un método específico
        
        Args:
            method_name: Nombre del método a plotear
            save_path: Ruta para guardar la figura
        """
        if method_name not in self.roc_data:
            raise ValueError(f"No hay datos ROC para el método: {method_name}")
        
        data = self.roc_data[method_name]
        
        plt.figure(figsize=(8, 6))
        plt.plot(data['fpr'], data['tpr'], linewidth=2, 
                label=f"{method_name} (AUC = {data['auc']:.3f})")
        plt.plot([0, 1], [0, 1], 'k--', label='Clasificador Aleatorio')
        
        plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
        plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad)')
        plt.title(f'Curva ROC - {method_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Curva ROC guardada en: {save_path}")
        
        show_plot_briefly(1.0)
    
    def plot_roc_comparison(self, method_names: Optional[List[str]] = None, 
                           save_path: Optional[str] = None):
        """
        Plotea comparación de curvas ROC
        
        Args:
            method_names: Lista de métodos a comparar (todos si es None)
            save_path: Ruta para guardar la figura
        """
        if method_names is None:
            method_names = list(self.roc_data.keys())
        
        plt.figure(figsize=(10, 8))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        
        for i, method_name in enumerate(method_names):
            if method_name not in self.roc_data:
                print(f"Advertencia: No hay datos para {method_name}")
                continue
            
            data = self.roc_data[method_name]
            color = colors[i % len(colors)]
            
            plt.plot(data['fpr'], data['tpr'], linewidth=2, color=color,
                    label=f"{method_name} (AUC = {data['auc']:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--', label='Clasificador Aleatorio')
        
        plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
        plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad)')
        plt.title('Comparación de Curvas ROC')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparación ROC guardada en: {save_path}")
        
        show_plot_briefly(1.0)
    
    def plot_threshold_analysis(self, y_true: np.ndarray, y_scores: np.ndarray,
                               method_name: str = "Unknown", save_path: Optional[str] = None):
        """
        Plotea análisis de umbrales con puntos óptimos marcados
        
        Args:
            y_true: Etiquetas verdaderas
            y_scores: Scores de decisión
            method_name: Nombre del método
            save_path: Ruta para guardar la figura
        """
        # Realizar análisis completo
        analysis = self.comprehensive_threshold_analysis(y_true, y_scores, method_name)
        
        # Obtener datos ROC
        data = self.roc_data[method_name]
        
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Curva ROC con puntos óptimos
        plt.subplot(2, 2, 1)
        plt.plot(data['fpr'], data['tpr'], 'b-', linewidth=2, label=f"ROC (AUC={data['auc']:.3f})")
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Aleatorio')
        
        # Marcar puntos óptimos
        youden_info = analysis['youden']
        eer_info = analysis['eer']
        
        plt.plot(1-youden_info['specificity'], youden_info['sensitivity'], 'ro', 
                markersize=8, label=f"Youden (J={youden_info['index']:.3f})")
        plt.plot(1-eer_info['specificity'], eer_info['sensitivity'], 'go', 
                markersize=8, label=f"EER ({eer_info['value']:.3f})")
        
        if analysis['high_sensitivity']['feasible']:
            hs_info = analysis['high_sensitivity']
            plt.plot(1-hs_info['specificity'], hs_info['sensitivity'], 'mo', 
                    markersize=8, label=f"Alta Sens. ({hs_info['sensitivity']:.3f})")
        
        plt.xlabel('FPR (1 - Especificidad)')
        plt.ylabel('TPR (Sensibilidad)')
        plt.title('Curva ROC con Puntos Óptimos')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Índice de Youden vs Umbral
        plt.subplot(2, 2, 2)
        youden_indices = data['tpr'] - data['fpr']
        plt.plot(data['thresholds'], youden_indices, 'b-', linewidth=2)
        plt.axvline(youden_info['threshold'], color='r', linestyle='--', 
                   label=f"Óptimo: {youden_info['threshold']:.3f}")
        plt.xlabel('Umbral')
        plt.ylabel('Índice de Youden (J)')
        plt.title('Índice de Youden vs Umbral')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Sensibilidad y Especificidad vs Umbral
        plt.subplot(2, 2, 3)
        plt.plot(data['thresholds'], data['tpr'], 'g-', linewidth=2, label='Sensibilidad')
        plt.plot(data['thresholds'], 1-data['fpr'], 'r-', linewidth=2, label='Especificidad')
        plt.axvline(eer_info['threshold'], color='b', linestyle='--', 
                   label=f"EER: {eer_info['threshold']:.3f}")
        plt.xlabel('Umbral')
        plt.ylabel('Métrica')
        plt.title('Sensibilidad y Especificidad vs Umbral')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Resumen de criterios
        plt.subplot(2, 2, 4)
        plt.axis('off')
        
        summary_text = f"""
        RESUMEN DE CRITERIOS DE SELECCIÓN
        
        Método: {method_name}
        AUC: {data['auc']:.4f}
        
        Índice de Youden:
        • Umbral: {youden_info['threshold']:.4f}
        • Índice J: {youden_info['index']:.4f}
        • Sensibilidad: {youden_info['sensitivity']:.4f}
        • Especificidad: {youden_info['specificity']:.4f}
        
        Equal Error Rate:
        • Umbral: {eer_info['threshold']:.4f}
        • EER: {eer_info['value']:.4f}
        • Sensibilidad: {eer_info['sensitivity']:.4f}
        • Especificidad: {eer_info['specificity']:.4f}
        
        Alta Sensibilidad:
        """
        
        if analysis['high_sensitivity']['feasible']:
            hs_info = analysis['high_sensitivity']
            summary_text += f"""• Factible: Sí
        • Umbral: {hs_info['threshold']:.4f}
        • Sensibilidad: {hs_info['sensitivity']:.4f}
        • Especificidad: {hs_info['specificity']:.4f}"""
        else:
            summary_text += f"""• Factible: No
        • Máx. sensibilidad: {analysis['high_sensitivity']['max_achievable_sensitivity']:.4f}"""
        
        plt.text(0.05, 0.95, summary_text, fontsize=9, verticalalignment='top',
                fontfamily='monospace', transform=plt.gca().transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Análisis de umbrales guardado en: {save_path}")
        
        show_plot_briefly(1.0)
        
        return analysis


def select_optimal_threshold(y_true: np.ndarray, y_scores: np.ndarray, 
                           criterion: str = 'youden') -> Tuple[float, Dict]:
    """
    Función de conveniencia para seleccionar umbral óptimo
    
    Args:
        y_true: Etiquetas verdaderas
        y_scores: Scores de decisión
        criterion: Criterio de selección ('youden', 'eer', 'high_sensitivity')
        
    Returns:
        Tupla con (umbral_óptimo, información_adicional)
    """
    analyzer = ROCAnalyzer()
    
    if criterion == 'youden':
        threshold, value, info = analyzer.find_youden_threshold(y_true, y_scores)
        return threshold, info
    elif criterion == 'eer':
        threshold, value, info = analyzer.find_eer_threshold(y_true, y_scores)
        return threshold, info
    elif criterion == 'high_sensitivity':
        threshold, info = analyzer.find_high_sensitivity_threshold(y_true, y_scores)
        return threshold, info
    else:
        raise ValueError("criterion debe ser 'youden', 'eer', o 'high_sensitivity'")


if __name__ == "__main__":
    # Ejemplo de uso con datos sintéticos
    np.random.seed(42)
    
    # Generar datos sintéticos con diferentes calidades de separación
    n_samples = 2000
    
    # Clase 0: scores bajos
    scores_class0 = np.random.normal(-1, 1, int(n_samples * 0.7))
    labels_class0 = np.zeros(len(scores_class0))
    
    # Clase 1: scores altos
    scores_class1 = np.random.normal(1.5, 1, int(n_samples * 0.3))
    labels_class1 = np.ones(len(scores_class1))
    
    # Combinar
    y_true = np.concatenate([labels_class0, labels_class1])
    y_scores = np.concatenate([scores_class0, scores_class1])
    
    # Mezclar
    indices = np.random.permutation(len(y_true))
    y_true = y_true[indices]
    y_scores = y_scores[indices]
    
    # Análisis completo
    analyzer = ROCAnalyzer()
    analysis = analyzer.comprehensive_threshold_analysis(
        y_true, y_scores, "Método Ejemplo"
    )
    
    print("Análisis de selección de umbrales:")
    print(f"AUC: {analysis['auc']:.4f}")
    print(f"\nYouden - Umbral: {analysis['youden']['threshold']:.4f}, J: {analysis['youden']['index']:.4f}")
    print(f"EER - Umbral: {analysis['eer']['threshold']:.4f}, EER: {analysis['eer']['value']:.4f}")
    print(f"Alta Sens. - Factible: {analysis['high_sensitivity']['feasible']}")
    
    # Visualizar análisis
    analyzer.plot_threshold_analysis(y_true, y_scores, "Método Ejemplo")