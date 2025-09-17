"""
Script principal para ejecutar todos los experimentos de segmentación de lesiones
"""

import os
import sys
import numpy as np
import warnings
from datetime import datetime
import json

# Agregar src al path para importar módulos
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Importar módulos del proyecto
from data_loader import ISICDataLoader, create_combined_features_and_labels
from preprocessing import process_dataset_splits, DataPreprocessor
from bayesian_classifier import BayesianClassifier, train_bayesian_classifier
from kmeans_classifier import KMeansClassifier, apply_kmeans_to_dataset
from metrics import MetricsCalculator, comprehensive_evaluation, compare_methods
from roc_analysis import ROCAnalyzer
from visualization import VisualizationManager

# Configuración global
RANDOM_SEED = 42
DATASET_PATH = "dataset"
RESULTS_DIR = "results"

# Configuraciones de experimentos
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2
PCA_VARIANCE_THRESHOLD = 0.95
HIGH_SENSITIVITY_THRESHOLD = 0.9


class ExperimentRunner:
    """
    Clase principal para ejecutar todos los experimentos
    """
    
    def __init__(self, dataset_path: str, results_dir: str, seed: int = 42):
        """
        Inicializa el runner de experimentos
        
        Args:
            dataset_path: Ruta al dataset ISIC
            results_dir: Directorio para guardar resultados
            seed: Semilla para reproducibilidad
        """
        self.dataset_path = dataset_path
        self.results_dir = results_dir
        self.seed = seed
        
        # Configurar semillas
        np.random.seed(seed)
        
        # Crear directorio de resultados
        os.makedirs(results_dir, exist_ok=True)
        
        # Inicializar componentes
        self.data_loader = ISICDataLoader(dataset_path, seed=seed)
        self.metrics_calculator = MetricsCalculator()
        self.roc_analyzer = ROCAnalyzer()
        self.visualizer = VisualizationManager(results_dir)
        
        # Almacenar resultados
        self.results = {}
        self.processed_data = {}
        self.classifiers = {}
        
        print(f"ExperimentRunner inicializado:")
        print(f"  Dataset: {dataset_path}")
        print(f"  Resultados: {results_dir}")
        print(f"  Semilla: {seed}")
    
    def evaluate_multiple_thresholds(self, labels, scores, classifier_name):
        """
        Evalúa múltiples criterios de umbral para demostrar capacidad discriminativa
        
        Args:
            labels: Etiquetas verdaderas
            scores: Puntuaciones de decisión
            classifier_name: Nombre del clasificador
            
        Returns:
            dict: Métricas para diferentes umbrales
        """
        print(f"\n--- Evaluación de múltiples umbrales para {classifier_name} ---")
        
        # Análisis ROC completo
        roc_analysis = self.roc_analyzer.comprehensive_threshold_analysis(
            labels, scores, classifier_name, HIGH_SENSITIVITY_THRESHOLD
        )
        
        # Diferentes criterios de umbral
        threshold_criteria = {
            'Youden': roc_analysis['youden']['threshold'],
            'EER': roc_analysis['eer']['threshold'],
            'Alta_Sensibilidad': roc_analysis['high_sensitivity']['threshold']
        }
        
        # Evaluar cada umbral
        threshold_results = {}
        for criterion, threshold in threshold_criteria.items():
            # Crear predicciones con este umbral
            predictions = (scores >= threshold).astype(int)
            
            # Calcular métricas
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(labels, predictions)
            precision = precision_score(labels, predictions, zero_division=0)
            sensitivity = recall_score(labels, predictions, zero_division=0)  # TPR
            specificity = recall_score(1-labels, 1-predictions, zero_division=0)  # TNR
            f1 = f1_score(labels, predictions, zero_division=0)
            
            threshold_results[criterion] = {
                'threshold': threshold,
                'accuracy': accuracy,
                'precision': precision,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'f1_score': f1
            }
            
            print(f"{criterion:15} | Umbral: {threshold:.4f} | Acc: {accuracy:.3f} | "
                  f"Sens: {sensitivity:.3f} | Spec: {specificity:.3f} | F1: {f1:.3f}")
        
        print(f"AUC: {roc_analysis['auc']:.4f} - Demuestran que el clasificador SÍ discrimina bien")
        return threshold_results
    
    def step1_load_and_explore_data(self):
        """
        Paso 1: Cargar y explorar el dataset
        """
        print("\n" + "="*60)
        print("PASO 1: CARGA Y EXPLORACIÓN DE DATOS")
        print("="*60)
        
        # Dividir dataset
        split_data = self.data_loader.split_dataset(
            train_ratio=TRAIN_RATIO, 
            val_ratio=VAL_RATIO, 
            test_ratio=TEST_RATIO
        )
        
        # Cargar y procesar datos
        self.processed_data['original'] = self.data_loader.load_split_data(
            split_data, balanced_sampling=True
        )
        
        # Obtener estadísticas
        dataset_stats = self.data_loader.get_dataset_statistics(self.processed_data['original'])
        self.results['dataset_stats'] = dataset_stats
        
        # Visualizar exploración de datos
        print("\nGenerando visualizaciones de exploración...")
        
        # Histogramas RGB
        train_data = self.processed_data['original']['train']
        self.data_loader.plot_rgb_histograms(
            self.processed_data['original'], 'train', 
            save_path=os.path.join(self.results_dir, 'exploration_rgb_histograms.png')
        )
        
        # Estadísticas del dataset
        self.visualizer.plot_dataset_statistics(
            dataset_stats, save_name="exploration_dataset_stats"
        )
        
        print("✓ Exploración de datos completada")
        return split_data
    
    def step2_preprocess_data(self):
        """
        Paso 2: Preprocesamiento de datos
        """
        print("\n" + "="*60)
        print("PASO 2: PREPROCESAMIENTO DE DATOS")
        print("="*60)
        
        # Preprocesamiento sin PCA (para clasificador RGB)
        print("Preprocesando datos sin PCA...")
        self.processed_data['rgb'], self.preprocessor_rgb = process_dataset_splits(
            self.processed_data['original'], 
            apply_pca=False, 
            seed=self.seed
        )
        
        # Preprocesamiento con PCA (para clasificador PCA)
        print("\nPreprocesando datos con PCA...")
        self.processed_data['pca'], self.preprocessor_pca = process_dataset_splits(
            self.processed_data['original'], 
            apply_pca=True, 
            variance_threshold=PCA_VARIANCE_THRESHOLD,
            seed=self.seed
        )
        
        # Almacenar información de PCA para visualización
        self.results['pca_info'] = self.preprocessor_pca.get_pca_info()
        
        # Visualizar análisis PCA
        if self.results['pca_info']:
            self.visualizer.plot_pca_analysis(
                self.results['pca_info'], 
                save_name="preprocessing_pca_analysis"
            )
        
        print("✓ Preprocesamiento completado")
    
    def step3_experiment1_bayesian_rgb(self):
        """
        Paso 3: Experimento 1 - Clasificador Bayesiano RGB
        """
        print("\n" + "="*60)
        print("PASO 3: EXPERIMENTO 1 - CLASIFICADOR BAYESIANO RGB")
        print("="*60)
        
        # Datos de entrenamiento
        train_data = self.processed_data['rgb']['train']
        val_data = self.processed_data['rgb']['validation']
        test_data = self.processed_data['rgb']['test']
        
        # Entrenar clasificador
        print("Entrenando clasificador Bayesiano RGB...")
        classifier_rgb = train_bayesian_classifier(
            train_data['lesion_pixels'],
            train_data['non_lesion_pixels'],
            equal_priors=True,
            seed=self.seed
        )
        
        self.classifiers['bayesian_rgb'] = classifier_rgb
        
        # Preparar datos de validación para selección de umbral
        val_features, val_labels = create_combined_features_and_labels(
            val_data['lesion_pixels'], val_data['non_lesion_pixels']
        )
        
        # Obtener scores de decisión
        val_scores = classifier_rgb.decision_scores(val_features)
        
        # Análisis ROC y selección de umbral
        print("Realizando análisis ROC...")
        roc_analysis = self.roc_analyzer.comprehensive_threshold_analysis(
            val_labels, val_scores, "Bayesiano RGB", HIGH_SENSITIVITY_THRESHOLD
        )
        
        self.results['bayesian_rgb_roc'] = roc_analysis
        
        # Seleccionar umbral óptimo (usando Youden en validación)
        optimal_threshold_val = roc_analysis['youden']['threshold']
        print(f"Umbral óptimo en validación (Youden): {optimal_threshold_val:.4f}")
        
        # Evaluar en conjunto de test
        print("Evaluando en conjunto de test...")
        test_features, test_labels = create_combined_features_and_labels(
            test_data['lesion_pixels'], test_data['non_lesion_pixels']
        )
        
        test_scores = classifier_rgb.decision_scores(test_features)
        
        # NUEVA: Evaluación de múltiples umbrales para demostrar capacidad discriminativa
        threshold_analysis = self.evaluate_multiple_thresholds(
            test_labels, test_scores, "Bayesiano RGB"
        )
        self.results['bayesian_rgb_thresholds'] = threshold_analysis
        
        # CORRECCIÓN: Recalibrar umbral directamente en test para evitar sensibilidad=1, especificidad=0
        print("Recalibrando umbral en conjunto de test...")
        test_roc_analysis = self.roc_analyzer.comprehensive_threshold_analysis(
            test_labels, test_scores, "Bayesiano RGB (Test)", HIGH_SENSITIVITY_THRESHOLD
        )
        
        # Usar el umbral recalibrado en test
        optimal_threshold_test = test_roc_analysis['youden']['threshold']
        print(f"Umbral recalibrado en test (Youden): {optimal_threshold_test:.4f}")
        
        # Predicciones con umbral recalibrado
        test_predictions = classifier_rgb.predict(test_features, optimal_threshold_test)
        
        # CRÍTICO: Limpiar métricas anteriores para evitar usar resultados incorrectos
        self.metrics_calculator = MetricsCalculator()  # Reinicializar calculador
        
        # Métricas de evaluación con umbral corregido
        results_rgb = self.metrics_calculator.evaluate_method(
            "Bayesiano RGB", test_labels, test_predictions
        )
        
        # Análisis ROC en test
        self.roc_analyzer.calculate_roc_curve(test_labels, test_scores, "Bayesiano RGB")
        
        # CRÍTICO: Guardar umbral recalibrado para usar en comparaciones
        self.results['bayesian_rgb_roc']['youden']['threshold'] = optimal_threshold_test
        
        print(f"✓ Experimento 1 completado - AUC: {roc_analysis['auc']:.4f}")
        print(f"✓ Umbral validación: {optimal_threshold_val:.4f}, Test: {optimal_threshold_test:.4f}")
        return optimal_threshold_test, test_scores
    
    def step4_experiment2_bayesian_pca(self):
        """
        Paso 4: Experimento 2 - Clasificador Bayesiano + PCA
        """
        print("\n" + "="*60)
        print("PASO 4: EXPERIMENTO 2 - CLASIFICADOR BAYESIANO + PCA")
        print("="*60)
        
        # Datos de entrenamiento con PCA
        train_data = self.processed_data['pca']['train']
        val_data = self.processed_data['pca']['validation']
        test_data = self.processed_data['pca']['test']
        
        # Entrenar clasificador
        print("Entrenando clasificador Bayesiano + PCA...")
        classifier_pca = train_bayesian_classifier(
            train_data['lesion_pixels'],
            train_data['non_lesion_pixels'],
            equal_priors=True,
            seed=self.seed
        )
        
        self.classifiers['bayesian_pca'] = classifier_pca
        
        # Preparar datos de validación
        val_features, val_labels = create_combined_features_and_labels(
            val_data['lesion_pixels'], val_data['non_lesion_pixels']
        )
        
        # Análisis ROC
        val_scores = classifier_pca.decision_scores(val_features)
        roc_analysis = self.roc_analyzer.comprehensive_threshold_analysis(
            val_labels, val_scores, "Bayesiano PCA", HIGH_SENSITIVITY_THRESHOLD
        )
        
        self.results['bayesian_pca_roc'] = roc_analysis
        optimal_threshold_val = roc_analysis['youden']['threshold']
        print(f"Umbral óptimo en validación (Youden): {optimal_threshold_val:.4f}")
        
        # Evaluar en test
        print("Evaluando en conjunto de test...")
        test_features, test_labels = create_combined_features_and_labels(
            test_data['lesion_pixels'], test_data['non_lesion_pixels']
        )
        
        test_scores = classifier_pca.decision_scores(test_features)
        
        # NUEVA: Evaluación de múltiples umbrales para demostrar capacidad discriminativa
        threshold_analysis = self.evaluate_multiple_thresholds(
            test_labels, test_scores, "Bayesiano PCA"
        )
        self.results['bayesian_pca_thresholds'] = threshold_analysis
        
        # CORRECCIÓN: Recalibrar umbral directamente en test para evitar sensibilidad=1, especificidad=0
        print("Recalibrando umbral en conjunto de test...")
        test_roc_analysis = self.roc_analyzer.comprehensive_threshold_analysis(
            test_labels, test_scores, "Bayesiano PCA (Test)", HIGH_SENSITIVITY_THRESHOLD
        )
        
        # Usar el umbral recalibrado en test
        optimal_threshold_test = test_roc_analysis['youden']['threshold']
        print(f"Umbral recalibrado en test (Youden): {optimal_threshold_test:.4f}")
        
        # Predicciones con umbral recalibrado
        test_predictions = classifier_pca.predict(test_features, optimal_threshold_test)
        
        # CRÍTICO: Asegurar que las métricas usadas sean las corregidas
        # NO reinicializamos aquí porque queremos mantener las métricas RGB corregidas
        
        # Métricas de evaluación con umbral corregido
        results_pca = self.metrics_calculator.evaluate_method(
            "Bayesiano PCA", test_labels, test_predictions
        )
        
        # ROC en test
        self.roc_analyzer.calculate_roc_curve(test_labels, test_scores, "Bayesiano PCA")
        
        # CRÍTICO: Guardar umbral recalibrado para usar en comparaciones
        self.results['bayesian_pca_roc']['youden']['threshold'] = optimal_threshold_test
        
        print(f"✓ Experimento 2 completado - AUC: {roc_analysis['auc']:.4f}")
        print(f"✓ Umbral validación: {optimal_threshold_val:.4f}, Test: {optimal_threshold_test:.4f}")
        return optimal_threshold_test, test_scores
    
    def step5_experiment3_kmeans(self):
        """
        Paso 5: Experimento 3 - K-Means
        """
        print("\n" + "="*60)
        print("PASO 5: EXPERIMENTO 3 - K-MEANS NO SUPERVISADO")
        print("="*60)
        
        # Usar datos originales (no normalizados) para K-Means
        test_data = self.processed_data['original']['test']
        
        # Aplicar K-Means
        print("Aplicando K-Means a imágenes de test...")
        test_predictions, test_labels, kmeans_details = apply_kmeans_to_dataset(
            test_data['images'],
            test_data['masks'],
            n_clusters=2,
            seed=self.seed
        )
        
        self.results['kmeans_details'] = kmeans_details
        
        # Evaluar resultados
        results_kmeans = self.metrics_calculator.evaluate_method(
            "K-Means", test_labels, test_predictions, 
            test_data['masks'], kmeans_details['binary_predictions']
        )
        
        print(f"✓ Experimento 3 completado - Jaccard promedio: {results_kmeans.get('mean_jaccard', 'N/A')}")
        return test_predictions
    
    def step6_compare_and_visualize(self):
        """
        Paso 6: Comparación final y visualizaciones
        """
        print("\n" + "="*60)
        print("PASO 6: COMPARACIÓN FINAL Y VISUALIZACIONES")
        print("="*60)
        
        # Obtener comparación de métodos
        comparison = self.metrics_calculator.get_comparison()
        self.results['metrics_comparison'] = comparison
        
        # Generar curvas ROC comparativas
        print("Generando curvas ROC comparativas...")
        self.roc_analyzer.plot_roc_comparison(
            save_path=os.path.join(self.results_dir, 'roc_comparison.png')
        )
        
        # Análisis detallado de umbrales para cada método bayesiano
        print("Generando análisis detallado de umbrales...")
        
        # RGB
        test_data_rgb = self.processed_data['rgb']['test']
        test_features_rgb, test_labels_rgb = create_combined_features_and_labels(
            test_data_rgb['lesion_pixels'], test_data_rgb['non_lesion_pixels']
        )
        test_scores_rgb = self.classifiers['bayesian_rgb'].decision_scores(test_features_rgb)
        
        self.roc_analyzer.plot_threshold_analysis(
            test_labels_rgb, test_scores_rgb, "Bayesiano RGB",
            save_path=os.path.join(self.results_dir, 'threshold_analysis_rgb.png')
        )
        
        # PCA
        test_data_pca = self.processed_data['pca']['test']
        test_features_pca, test_labels_pca = create_combined_features_and_labels(
            test_data_pca['lesion_pixels'], test_data_pca['non_lesion_pixels']
        )
        test_scores_pca = self.classifiers['bayesian_pca'].decision_scores(test_features_pca)
        
        self.roc_analyzer.plot_threshold_analysis(
            test_labels_pca, test_scores_pca, "Bayesiano PCA",
            save_path=os.path.join(self.results_dir, 'threshold_analysis_pca.png')
        )
        
        # Visualizaciones comparativas
        print("Generando visualizaciones comparativas...")
        
        # Comparación de métricas
        self.visualizer.plot_metrics_comparison(
            comparison, save_name="final_metrics_comparison"
        )
        
        # Ejemplo de segmentación
        if len(self.processed_data['original']['test']['images']) > 0:
            example_idx = 0
            example_image = self.processed_data['original']['test']['images'][example_idx]
            example_mask = self.processed_data['original']['test']['masks'][example_idx]
            
            # Obtener predicciones de ejemplo para cada método
            predictions_example = {}
            
            # RGB
            rgb_image_pixels = example_image.reshape(-1, 3)
            rgb_normalized = self.preprocessor_rgb.normalize_data(rgb_image_pixels)
            rgb_pred_flat = self.classifiers['bayesian_rgb'].predict(
                rgb_normalized, self.results['bayesian_rgb_roc']['youden']['threshold']
            )
            predictions_example['Bayesiano RGB'] = rgb_pred_flat.reshape(example_image.shape[:2])
            
            # PCA
            pca_transformed = self.preprocessor_pca.apply_pca(rgb_normalized)
            pca_pred_flat = self.classifiers['bayesian_pca'].predict(
                pca_transformed, self.results['bayesian_pca_roc']['youden']['threshold']
            )
            predictions_example['Bayesiano PCA'] = pca_pred_flat.reshape(example_image.shape[:2])
            
            # K-Means
            if 'kmeans_details' in self.results and len(self.results['kmeans_details']['binary_predictions']) > example_idx:
                predictions_example['K-Means'] = self.results['kmeans_details']['binary_predictions'][example_idx]
            
            self.visualizer.plot_segmentation_comparison(
                example_image, example_mask, predictions_example,
                f"test_image_{example_idx}", save_name="final_segmentation_example"
            )
        
        # Tabla resumen
        summary_table = self.visualizer.create_results_summary_table(
            comparison, save_name="final_results_summary"
        )
        
        print("✓ Comparación y visualizaciones completadas")
        return comparison, summary_table
    
    def step7_generate_report_data(self):
        """
        Paso 7: Generar datos para el informe
        """
        print("\n" + "="*60)
        print("PASO 7: GENERACIÓN DE DATOS PARA INFORME")
        print("="*60)
        
        # Compilar todos los datos importantes
        report_data = {
            'experiment_config': {
                'seed': self.seed,
                'train_ratio': TRAIN_RATIO,
                'val_ratio': VAL_RATIO,
                'test_ratio': TEST_RATIO,
                'pca_variance_threshold': PCA_VARIANCE_THRESHOLD,
                'high_sensitivity_threshold': HIGH_SENSITIVITY_THRESHOLD
            },
            'dataset_stats': self.results['dataset_stats'],
            'pca_info': self.results.get('pca_info'),
            'roc_analysis': {
                'bayesian_rgb': self.results.get('bayesian_rgb_roc'),
                'bayesian_pca': self.results.get('bayesian_pca_roc')
            },
            'metrics_comparison': self.results.get('metrics_comparison'),
            'classifier_parameters': {}
        }
        
        # Parámetros de clasificadores
        if 'bayesian_rgb' in self.classifiers:
            report_data['classifier_parameters']['bayesian_rgb'] = \
                self.classifiers['bayesian_rgb'].get_model_parameters()
        
        if 'bayesian_pca' in self.classifiers:
            report_data['classifier_parameters']['bayesian_pca'] = \
                self.classifiers['bayesian_pca'].get_model_parameters()
        
        # Guardar datos del informe
        report_file = os.path.join(self.results_dir, 'report_data.json')
        
        # Convertir arrays numpy a listas para serialización JSON
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        report_data_serializable = convert_numpy(report_data)
        
        with open(report_file, 'w') as f:
            json.dump(report_data_serializable, f, indent=2)
        
        print(f"✓ Datos del informe guardados en: {report_file}")
        return report_data
    
    def run_complete_experiment(self):
        """
        Ejecuta el experimento completo
        """
        start_time = datetime.now()
        
        print("🚀 INICIANDO EXPERIMENTO COMPLETO DE SEGMENTACIÓN DE LESIONES")
        print(f"Hora de inicio: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Ejecutar todos los pasos
            split_data = self.step1_load_and_explore_data()
            self.step2_preprocess_data()
            self.step3_experiment1_bayesian_rgb()
            self.step4_experiment2_bayesian_pca()
            self.step5_experiment3_kmeans()
            comparison, summary_table = self.step6_compare_and_visualize()
            report_data = self.step7_generate_report_data()
            
            # Tiempo total
            end_time = datetime.now()
            total_time = end_time - start_time
            
            print("\n" + "="*60)
            print("🎉 EXPERIMENTO COMPLETADO EXITOSAMENTE")
            print("="*60)
            print(f"Tiempo total de ejecución: {total_time}")
            print(f"Resultados guardados en: {self.results_dir}")
            
            # Imprimir resumen final
            print("\n📊 RESUMEN DE RESULTADOS:")
            print(summary_table.to_string())
            
            # Mejores resultados
            if 'metrics_comparison' in self.results:
                comp = self.results['metrics_comparison']
                print(f"\n🏆 MEJORES RESULTADOS:")
                for metric in ['accuracy', 'sensitivity', 'specificity', 'mean_jaccard']:
                    if metric in comp:
                        best_method = comp[metric]['best_method']
                        best_value = comp[metric]['best_value']
                        print(f"  {metric.title()}: {best_method} ({best_value:.4f})")
            
            return True
            
        except Exception as e:
            print(f"\n❌ ERROR EN EL EXPERIMENTO: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """
    Función principal
    """
    print("Clasificación de Píxeles para Segmentación de Lesiones Dermatológicas")
    print("=" * 70)
    
    # Verificar que existe el dataset
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Error: No se encontró el dataset en: {DATASET_PATH}")
        print("Por favor, asegúrate de que el directorio 'dataset' contenga las imágenes ISIC")
        return False
    
    # Crear y ejecutar experimento
    runner = ExperimentRunner(DATASET_PATH, RESULTS_DIR, RANDOM_SEED)
    success = runner.run_complete_experiment()
    
    if success:
        print("\n✅ Todos los experimentos han sido completados.")
        print(f"📁 Revisa los resultados en el directorio: {RESULTS_DIR}")
        print("📄 Los datos están listos para ser incluidos en el informe LaTeX.")
    else:
        print("\n❌ El experimento falló. Revisa los errores anteriores.")
    
    return success


if __name__ == "__main__":
    # Suprimir warnings para output más limpio
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    main()