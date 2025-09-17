"""
Módulo de métricas de evaluación para clasificación de lesiones
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import warnings


def calculate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calcula matriz de confusión
    
    Args:
        y_true: Etiquetas verdaderas (N,)
        y_pred: Predicciones (N,)
        
    Returns:
        Matriz de confusión 2x2
    """
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def calculate_pixel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcula métricas de evaluación a nivel de píxel
    
    Args:
        y_true: Etiquetas verdaderas (N,)
        y_pred: Predicciones (N,)
        
    Returns:
        Diccionario con métricas
    """
    # Obtener matriz de confusión
    cm = calculate_confusion_matrix(y_true, y_pred)
    
    # Extraer valores
    tn, fp, fn, tp = cm.ravel()
    
    # Evitar división por cero
    epsilon = 1e-10
    
    # Calcular métricas
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    sensitivity = tp / (tp + fn + epsilon)  # También llamado recall o TPR
    specificity = tn / (tn + fp + epsilon)  # También llamado TNR
    
    # Métricas adicionales
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity + epsilon)
    balanced_accuracy = (sensitivity + specificity) / 2
    
    # False rates
    false_positive_rate = fp / (fp + tn + epsilon)
    false_negative_rate = fn / (fn + tp + epsilon)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1_score': f1_score,
        'balanced_accuracy': balanced_accuracy,
        'false_positive_rate': false_positive_rate,
        'false_negative_rate': false_negative_rate,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }


def jaccard_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula el índice de Jaccard (Intersection over Union)
    
    Args:
        y_true: Máscara verdadera binaria
        y_pred: Máscara predicha binaria
        
    Returns:
        Índice de Jaccard [0, 1]
    """
    # Convertir a booleano para operaciones lógicas
    y_true_bool = y_true.astype(bool)
    y_pred_bool = y_pred.astype(bool)
    
    # Calcular intersección y unión
    intersection = np.logical_and(y_true_bool, y_pred_bool).sum()
    union = np.logical_or(y_true_bool, y_pred_bool).sum()
    
    # Manejar caso especial donde ambas máscaras están vacías
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union


def dice_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula el coeficiente de Dice
    
    Args:
        y_true: Máscara verdadera binaria
        y_pred: Máscara predicha binaria
        
    Returns:
        Coeficiente de Dice [0, 1]
    """
    y_true_bool = y_true.astype(bool)
    y_pred_bool = y_pred.astype(bool)
    
    intersection = np.logical_and(y_true_bool, y_pred_bool).sum()
    total = y_true_bool.sum() + y_pred_bool.sum()
    
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return (2.0 * intersection) / total


def calculate_image_level_metrics(y_true_masks: List[np.ndarray], 
                                y_pred_masks: List[np.ndarray]) -> Dict[str, float]:
    """
    Calcula métricas a nivel de imagen
    
    Args:
        y_true_masks: Lista de máscaras verdaderas
        y_pred_masks: Lista de máscaras predichas
        
    Returns:
        Diccionario con métricas agregadas
    """
    if len(y_true_masks) != len(y_pred_masks):
        raise ValueError("Debe haber igual número de máscaras verdaderas y predichas")
    
    jaccard_scores = []
    dice_scores = []
    
    for y_true, y_pred in zip(y_true_masks, y_pred_masks):
        jaccard_scores.append(jaccard_index(y_true, y_pred))
        dice_scores.append(dice_coefficient(y_true, y_pred))
    
    return {
        'mean_jaccard': np.mean(jaccard_scores),
        'std_jaccard': np.std(jaccard_scores),
        'median_jaccard': np.median(jaccard_scores),
        'min_jaccard': np.min(jaccard_scores),
        'max_jaccard': np.max(jaccard_scores),
        'mean_dice': np.mean(dice_scores),
        'std_dice': np.std(dice_scores),
        'individual_jaccard': jaccard_scores,
        'individual_dice': dice_scores
    }


def comprehensive_evaluation(y_true: np.ndarray, y_pred: np.ndarray, 
                           y_true_masks: Optional[List[np.ndarray]] = None,
                           y_pred_masks: Optional[List[np.ndarray]] = None,
                           method_name: str = "Unknown") -> Dict:
    """
    Evaluación completa con métricas a nivel de píxel e imagen
    
    Args:
        y_true: Etiquetas verdaderas a nivel de píxel
        y_pred: Predicciones a nivel de píxel
        y_true_masks: Máscaras verdaderas por imagen (opcional)
        y_pred_masks: Máscaras predichas por imagen (opcional)
        method_name: Nombre del método evaluado
        
    Returns:
        Diccionario con todas las métricas
    """
    results = {
        'method_name': method_name,
        'n_samples': len(y_true)
    }
    
    # Métricas a nivel de píxel
    pixel_metrics = calculate_pixel_metrics(y_true, y_pred)
    results.update(pixel_metrics)
    
    # Métricas a nivel de imagen (si se proporcionan)
    if y_true_masks is not None and y_pred_masks is not None:
        image_metrics = calculate_image_level_metrics(y_true_masks, y_pred_masks)
        results.update(image_metrics)
        results['n_images'] = len(y_true_masks)
    
    return results


def compare_methods(results_list: List[Dict], 
                   metrics_to_compare: List[str] = None) -> Dict:
    """
    Compara múltiples métodos de clasificación
    
    Args:
        results_list: Lista de diccionarios con resultados de cada método
        metrics_to_compare: Lista de métricas a comparar
        
    Returns:
        Diccionario con comparación de métodos
    """
    if metrics_to_compare is None:
        metrics_to_compare = ['accuracy', 'precision', 'sensitivity', 'specificity', 
                            'f1_score', 'mean_jaccard']
    
    comparison = {}
    
    # Extraer nombres de métodos
    method_names = [result['method_name'] for result in results_list]
    comparison['methods'] = method_names
    
    # Comparar cada métrica
    for metric in metrics_to_compare:
        if all(metric in result for result in results_list):
            values = [result[metric] for result in results_list]
            comparison[metric] = {
                'values': values,
                'best_method': method_names[np.argmax(values)],
                'best_value': np.max(values),
                'worst_method': method_names[np.argmin(values)],
                'worst_value': np.min(values),
                'mean': np.mean(values),
                'std': np.std(values)
            }
    
    return comparison


def print_evaluation_results(results: Dict, detailed: bool = True):
    """
    Imprime resultados de evaluación de forma organizada
    
    Args:
        results: Diccionario con resultados de evaluación
        detailed: Si mostrar detalles adicionales
    """
    print(f"\n{'='*50}")
    print(f"RESULTADOS DE EVALUACIÓN: {results['method_name']}")
    print(f"{'='*50}")
    
    print(f"\n📊 MÉTRICAS A NIVEL DE PÍXEL")
    print(f"  Muestras evaluadas: {results['n_samples']:,}")
    print(f"  Exactitud:          {results['accuracy']:.4f}")
    print(f"  Precisión:          {results['precision']:.4f}")
    print(f"  Sensibilidad:       {results['sensitivity']:.4f}")
    print(f"  Especificidad:      {results['specificity']:.4f}")
    print(f"  F1-Score:           {results['f1_score']:.4f}")
    print(f"  Exactitud balanceada: {results['balanced_accuracy']:.4f}")
    
    if detailed:
        print(f"\n📈 MATRIZ DE CONFUSIÓN")
        print(f"  Verdaderos Positivos:  {results['true_positives']:,}")
        print(f"  Verdaderos Negativos:  {results['true_negatives']:,}")
        print(f"  Falsos Positivos:      {results['false_positives']:,}")
        print(f"  Falsos Negativos:      {results['false_negatives']:,}")
        print(f"  Tasa Falsos Positivos: {results['false_positive_rate']:.4f}")
        print(f"  Tasa Falsos Negativos: {results['false_negative_rate']:.4f}")
    
    # Métricas a nivel de imagen si están disponibles
    if 'mean_jaccard' in results:
        print(f"\n🖼️  MÉTRICAS A NIVEL DE IMAGEN")
        if 'n_images' in results:
            print(f"  Imágenes evaluadas:     {results['n_images']}")
        print(f"  Jaccard promedio:       {results['mean_jaccard']:.4f}")
        print(f"  Jaccard desv. est.:     {results['std_jaccard']:.4f}")
        print(f"  Jaccard mediana:        {results['median_jaccard']:.4f}")
        print(f"  Jaccard mín/máx:        {results['min_jaccard']:.4f} / {results['max_jaccard']:.4f}")
        
        if detailed and 'mean_dice' in results:
            print(f"  Dice promedio:          {results['mean_dice']:.4f}")
            print(f"  Dice desv. est.:        {results['std_dice']:.4f}")


def print_method_comparison(comparison: Dict):
    """
    Imprime comparación entre múltiples métodos
    
    Args:
        comparison: Resultado de compare_methods()
    """
    print(f"\n{'='*60}")
    print(f"COMPARACIÓN DE MÉTODOS")
    print(f"{'='*60}")
    
    methods = comparison['methods']
    print(f"Métodos comparados: {', '.join(methods)}")
    
    print(f"\n{'Métrica':<20} {'Mejor Método':<15} {'Valor':<10} {'Promedio':<10}")
    print(f"{'-'*60}")
    
    metrics_order = ['accuracy', 'precision', 'sensitivity', 'specificity', 
                    'f1_score', 'mean_jaccard']
    
    for metric in metrics_order:
        if metric in comparison:
            info = comparison[metric]
            print(f"{metric:<20} {info['best_method']:<15} {info['best_value']:<10.4f} {info['mean']:<10.4f}")


class MetricsCalculator:
    """
    Clase para cálculo y almacenamiento de métricas
    """
    
    def __init__(self):
        self.results = {}
    
    def evaluate_method(self, method_name: str, y_true: np.ndarray, y_pred: np.ndarray,
                       y_true_masks: Optional[List[np.ndarray]] = None,
                       y_pred_masks: Optional[List[np.ndarray]] = None) -> Dict:
        """
        Evalúa un método y almacena resultados
        
        Args:
            method_name: Nombre del método
            y_true: Etiquetas verdaderas
            y_pred: Predicciones
            y_true_masks: Máscaras verdaderas (opcional)
            y_pred_masks: Máscaras predichas (opcional)
            
        Returns:
            Diccionario con resultados
        """
        results = comprehensive_evaluation(
            y_true, y_pred, y_true_masks, y_pred_masks, method_name
        )
        
        self.results[method_name] = results
        return results
    
    def get_comparison(self, methods: Optional[List[str]] = None) -> Dict:
        """
        Obtiene comparación entre métodos almacenados
        
        Args:
            methods: Lista de métodos a comparar (todos si es None)
            
        Returns:
            Diccionario con comparación
        """
        if methods is None:
            methods = list(self.results.keys())
        
        results_list = [self.results[method] for method in methods if method in self.results]
        return compare_methods(results_list)
    
    def print_all_results(self, detailed: bool = True):
        """Imprime resultados de todos los métodos evaluados"""
        for method_name, results in self.results.items():
            print_evaluation_results(results, detailed)
    
    def export_results_to_dict(self) -> Dict:
        """Exporta todos los resultados a un diccionario"""
        return self.results.copy()


if __name__ == "__main__":
    # Ejemplo de uso con datos sintéticos
    np.random.seed(42)
    
    # Generar datos sintéticos
    n_samples = 10000
    y_true = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    
    # Simular predicciones de diferentes calidades
    # Método 1: Buena precisión
    y_pred1 = y_true.copy()
    noise_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    y_pred1[noise_indices] = 1 - y_pred1[noise_indices]
    
    # Método 2: Precisión media
    y_pred2 = y_true.copy()
    noise_indices = np.random.choice(n_samples, size=int(0.2 * n_samples), replace=False)
    y_pred2[noise_indices] = 1 - y_pred2[noise_indices]
    
    # Evaluar métodos
    calculator = MetricsCalculator()
    
    results1 = calculator.evaluate_method("Método Alto Rendimiento", y_true, y_pred1)
    results2 = calculator.evaluate_method("Método Medio Rendimiento", y_true, y_pred2)
    
    # Imprimir resultados
    calculator.print_all_results(detailed=True)
    
    # Comparar métodos
    comparison = calculator.get_comparison()
    print_method_comparison(comparison)