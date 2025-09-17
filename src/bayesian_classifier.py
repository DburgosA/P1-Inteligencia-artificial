"""
Implementación del Clasificador Bayesiano para segmentación de lesiones
"""

import numpy as np
from typing import Tuple, Optional, Dict
from scipy.stats import multivariate_normal
from scipy.linalg import det, inv
import warnings


class BayesianClassifier:
    """
    Clasificador Bayesiano basado en distribuciones gaussianas multivariadas
    """
    
    def __init__(self, seed: int = 42):
        """
        Inicializa el clasificador
        
        Args:
            seed: Semilla para reproducibilidad
        """
        self.seed = seed
        np.random.seed(seed)
        
        # Parámetros de las distribuciones
        self.mu_lesion = None       # Media de la clase lesión
        self.sigma_lesion = None    # Covarianza de la clase lesión
        self.mu_non_lesion = None   # Media de la clase no-lesión
        self.sigma_non_lesion = None # Covarianza de la clase no-lesión
        
        # Probabilidades prior
        self.prior_lesion = None
        self.prior_non_lesion = None
        
        # Estado del modelo
        self.is_fitted = False
        self.feature_dim = None
        
        # Para regularización de matrices de covarianza
        self.regularization = 1e-6
    
    def _estimate_gaussian_parameters(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estima parámetros de una distribución gaussiana multivariada
        
        Args:
            data: Datos de entrenamiento (N, D)
            
        Returns:
            Tupla con media (D,) y matriz de covarianza (D, D)
        """
        if len(data) == 0:
            raise ValueError("No hay datos para estimar parámetros")
        
        # Estimar media
        mu = np.mean(data, axis=0)
        
        # Estimar matriz de covarianza
        if len(data) == 1:
            # Solo una muestra, usar matriz identidad regularizada
            sigma = np.eye(data.shape[1]) * self.regularization
        else:
            # Estimación por máxima verosimilitud
            sigma = np.cov(data, rowvar=False)
            
            # Regularización para evitar matrices singulares
            sigma += np.eye(sigma.shape[0]) * self.regularization
        
        return mu, sigma
    
    def fit(self, lesion_data: np.ndarray, non_lesion_data: np.ndarray, 
           equal_priors: bool = True) -> 'BayesianClassifier':
        """
        Entrena el clasificador con datos de ambas clases
        
        Args:
            lesion_data: Datos de píxeles de lesión (N1, D)
            non_lesion_data: Datos de píxeles de no-lesión (N2, D)
            equal_priors: Si usar probabilidades prior iguales
            
        Returns:
            Self para method chaining
        """
        if len(lesion_data) == 0 or len(non_lesion_data) == 0:
            raise ValueError("Ambas clases deben tener al menos una muestra")
        
        # Verificar dimensiones
        if lesion_data.shape[1] != non_lesion_data.shape[1]:
            raise ValueError("Las dimensiones de ambas clases deben ser iguales")
        
        self.feature_dim = lesion_data.shape[1]
        
        # Estimar parámetros para cada clase
        self.mu_lesion, self.sigma_lesion = self._estimate_gaussian_parameters(lesion_data)
        self.mu_non_lesion, self.sigma_non_lesion = self._estimate_gaussian_parameters(non_lesion_data)
        
        # Estimar probabilidades prior
        if equal_priors:
            self.prior_lesion = 0.5
            self.prior_non_lesion = 0.5
        else:
            total_samples = len(lesion_data) + len(non_lesion_data)
            self.prior_lesion = len(lesion_data) / total_samples
            self.prior_non_lesion = len(non_lesion_data) / total_samples
        
        self.is_fitted = True
        
        # Imprimir información del modelo entrenado
        print("Clasificador Bayesiano entrenado:")
        print(f"  Dimensión de características: {self.feature_dim}")
        print(f"  Muestras de lesión: {len(lesion_data):,}")
        print(f"  Muestras de no-lesión: {len(non_lesion_data):,}")
        print(f"  Prior lesión: {self.prior_lesion:.3f}")
        print(f"  Prior no-lesión: {self.prior_non_lesion:.3f}")
        print(f"  Media lesión: {self.mu_lesion}")
        print(f"  Media no-lesión: {self.mu_non_lesion}")
        
        return self
    
    def _gaussian_pdf(self, x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
        """
        Calcula la densidad de probabilidad gaussiana multivariada
        
        Args:
            x: Punto a evaluar (D,)
            mu: Media (D,)
            sigma: Matriz de covarianza (D, D)
            
        Returns:
            Densidad de probabilidad
        """
        try:
            return multivariate_normal.pdf(x, mu, sigma)
        except np.linalg.LinAlgError:
            # Manejo de matrices singulares
            warnings.warn("Matriz de covarianza singular, usando regularización")
            sigma_reg = sigma + np.eye(sigma.shape[0]) * self.regularization * 10
            return multivariate_normal.pdf(x, mu, sigma_reg)
    
    def likelihood_ratio(self, x: np.ndarray) -> np.ndarray:
        """
        Calcula la razón de verosimilitudes para cada muestra
        
        Args:
            x: Muestras a evaluar (N, D)
            
        Returns:
            Razón de verosimilitudes (N,)
        """
        if not self.is_fitted:
            raise ValueError("El clasificador debe ser entrenado primero")
        
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        if x.shape[1] != self.feature_dim:
            raise ValueError(f"Dimensión incorrecta. Esperado: {self.feature_dim}, recibido: {x.shape[1]}")
        
        ratios = np.zeros(len(x))
        
        for i, sample in enumerate(x):
            # Calcular verosimilitudes
            likelihood_lesion = self._gaussian_pdf(sample, self.mu_lesion, self.sigma_lesion)
            likelihood_non_lesion = self._gaussian_pdf(sample, self.mu_non_lesion, self.sigma_non_lesion)
            
            # Evitar división por cero
            if likelihood_non_lesion == 0:
                ratios[i] = np.inf if likelihood_lesion > 0 else 1.0
            else:
                ratios[i] = likelihood_lesion / likelihood_non_lesion
        
        return ratios
    
    def likelihood_ratio_vectorized(self, x: np.ndarray) -> np.ndarray:
        """
        Versión vectorizada de likelihood_ratio para mejor rendimiento
        
        Args:
            x: Muestras a evaluar (N, D)
            
        Returns:
            Razón de verosimilitudes (N,)
        """
        if not self.is_fitted:
            raise ValueError("El clasificador debe ser entrenado primero")
        
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        if x.shape[1] != self.feature_dim:
            raise ValueError(f"Dimensión incorrecta. Esperado: {self.feature_dim}, recibido: {x.shape[1]}")
        
        # Calcular diferencias
        diff_lesion = x - self.mu_lesion
        diff_non_lesion = x - self.mu_non_lesion
        
        # Calcular log-likelihoods vectorizados (asumiendo matrices diagonales)
        log_likelihood_lesion = -0.5 * np.sum(diff_lesion**2 / np.diag(self.sigma_lesion), axis=1)
        log_likelihood_non_lesion = -0.5 * np.sum(diff_non_lesion**2 / np.diag(self.sigma_non_lesion), axis=1)
        
        # Calcular determinantes (para matrices diagonales es el producto de elementos diagonales)
        det_lesion = np.prod(np.diag(self.sigma_lesion))
        det_non_lesion = np.prod(np.diag(self.sigma_non_lesion))
        
        # Agregar términos de normalización
        log_likelihood_lesion -= 0.5 * np.log(det_lesion)
        log_likelihood_non_lesion -= 0.5 * np.log(det_non_lesion)
        
        # Calcular ratios en espacio log para estabilidad
        log_ratios = log_likelihood_lesion - log_likelihood_non_lesion
        
        # Convertir a ratio normal
        ratios = np.exp(log_ratios)
        
        # Manejar casos extremos
        ratios[np.isnan(ratios)] = 1.0
        ratios[np.isinf(ratios)] = 1e10
        
        return ratios
    
    def posterior_probabilities(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula probabilidades posteriores para ambas clases
        
        Args:
            x: Muestras a evaluar (N, D)
            
        Returns:
            Tupla con P(lesión|x) y P(no-lesión|x)
        """
        if not self.is_fitted:
            raise ValueError("El clasificador debe ser entrenado primero")
        
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        post_lesion = np.zeros(len(x))
        post_non_lesion = np.zeros(len(x))
        
        for i, sample in enumerate(x):
            # Calcular verosimilitudes
            likelihood_lesion = self._gaussian_pdf(sample, self.mu_lesion, self.sigma_lesion)
            likelihood_non_lesion = self._gaussian_pdf(sample, self.mu_non_lesion, self.sigma_non_lesion)
            
            # Calcular evidencia (denominador)
            evidence = (likelihood_lesion * self.prior_lesion + 
                       likelihood_non_lesion * self.prior_non_lesion)
            
            # Evitar división por cero
            if evidence == 0:
                post_lesion[i] = 0.5
                post_non_lesion[i] = 0.5
            else:
                post_lesion[i] = (likelihood_lesion * self.prior_lesion) / evidence
                post_non_lesion[i] = (likelihood_non_lesion * self.prior_non_lesion) / evidence
        
        return post_lesion, post_non_lesion
    
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """
        Predice probabilidades de clase para las muestras
        
        Args:
            x: Muestras a evaluar (N, D)
            
        Returns:
            Probabilidades (N, 2) donde columna 0 = no-lesión, columna 1 = lesión
        """
        post_lesion, post_non_lesion = self.posterior_probabilities(x)
        return np.column_stack([post_non_lesion, post_lesion])
    
    def predict(self, x: np.ndarray, threshold: float = 1.0) -> np.ndarray:
        """
        Predice clases basado en razón de verosimilitudes
        
        Args:
            x: Muestras a evaluar (N, D)
            threshold: Umbral para la razón de verosimilitudes
            
        Returns:
            Predicciones (N,) donde 1 = lesión, 0 = no-lesión
        """
        ratios = self.likelihood_ratio(x)
        return (ratios > threshold).astype(int)
    
    def decision_scores(self, x: np.ndarray) -> np.ndarray:
        """
        Calcula scores de decisión (log razón de verosimilitudes) - VERSIÓN OPTIMIZADA
        
        Args:
            x: Muestras a evaluar (N, D)
            
        Returns:
            Scores de decisión (N,)
        """
        ratios = self.likelihood_ratio_vectorized(x)
        # Usar log para estabilidad numérica
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            log_ratios = np.log(ratios)
            
        # Manejar casos extremos
        log_ratios[np.isinf(log_ratios) & (log_ratios > 0)] = 50  # log(ratio muy grande)
        log_ratios[np.isinf(log_ratios) & (log_ratios < 0)] = -50  # log(ratio muy pequeño)
        log_ratios[np.isnan(log_ratios)] = 0  # casos ambiguos
        
        return log_ratios
    
    def get_model_parameters(self) -> Dict:
        """
        Obtiene los parámetros del modelo entrenado
        
        Returns:
            Diccionario con parámetros del modelo
        """
        if not self.is_fitted:
            raise ValueError("El clasificador debe ser entrenado primero")
        
        return {
            'mu_lesion': self.mu_lesion.copy(),
            'sigma_lesion': self.sigma_lesion.copy(),
            'mu_non_lesion': self.mu_non_lesion.copy(),
            'sigma_non_lesion': self.sigma_non_lesion.copy(),
            'prior_lesion': self.prior_lesion,
            'prior_non_lesion': self.prior_non_lesion,
            'feature_dim': self.feature_dim
        }
    
    def mahalanobis_distance(self, x: np.ndarray, class_label: str) -> np.ndarray:
        """
        Calcula distancia de Mahalanobis a una clase específica
        
        Args:
            x: Muestras a evaluar (N, D)
            class_label: 'lesion' o 'non_lesion'
            
        Returns:
            Distancias de Mahalanobis (N,)
        """
        if not self.is_fitted:
            raise ValueError("El clasificador debe ser entrenado primero")
        
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        if class_label == 'lesion':
            mu = self.mu_lesion
            sigma = self.sigma_lesion
        elif class_label == 'non_lesion':
            mu = self.mu_non_lesion
            sigma = self.sigma_non_lesion
        else:
            raise ValueError("class_label debe ser 'lesion' o 'non_lesion'")
        
        try:
            sigma_inv = inv(sigma)
        except np.linalg.LinAlgError:
            # Usar pseudoinversa si la matriz es singular
            sigma_inv = np.linalg.pinv(sigma)
        
        distances = np.zeros(len(x))
        for i, sample in enumerate(x):
            diff = sample - mu
            distances[i] = np.sqrt(diff.T @ sigma_inv @ diff)
        
        return distances


def train_bayesian_classifier(lesion_data: np.ndarray, non_lesion_data: np.ndarray,
                             equal_priors: bool = True, seed: int = 42) -> BayesianClassifier:
    """
    Función de conveniencia para entrenar un clasificador Bayesiano
    
    Args:
        lesion_data: Datos de píxeles de lesión
        non_lesion_data: Datos de píxeles de no-lesión
        equal_priors: Si usar probabilidades prior iguales
        seed: Semilla para reproducibilidad
        
    Returns:
        Clasificador entrenado
    """
    classifier = BayesianClassifier(seed=seed)
    classifier.fit(lesion_data, non_lesion_data, equal_priors=equal_priors)
    return classifier


if __name__ == "__main__":
    # Ejemplo de uso con datos sintéticos
    np.random.seed(42)
    
    # Generar datos sintéticos
    n_samples = 1000
    
    # Clase lesión: media [100, 50, 75], covarianza con correlación
    mu_lesion = np.array([100, 50, 75])
    sigma_lesion = np.array([[400, 50, 25], [50, 300, 0], [25, 0, 200]])
    lesion_synthetic = np.random.multivariate_normal(mu_lesion, sigma_lesion, n_samples)
    
    # Clase no-lesión: media [150, 100, 125], covarianza diferente
    mu_non_lesion = np.array([150, 100, 125])
    sigma_non_lesion = np.array([[300, -25, 50], [-25, 400, 25], [50, 25, 250]])
    non_lesion_synthetic = np.random.multivariate_normal(mu_non_lesion, sigma_non_lesion, n_samples)
    
    # Entrenar clasificador
    print("Entrenando clasificador con datos sintéticos...")
    classifier = train_bayesian_classifier(lesion_synthetic, non_lesion_synthetic)
    
    # Probar predicciones
    test_data = np.array([[100, 50, 75], [150, 100, 125], [125, 75, 100]])
    
    print(f"\nProbando con datos de prueba:")
    print(f"Datos de prueba: {test_data}")
    
    # Razones de verosimilitud
    ratios = classifier.likelihood_ratio(test_data)
    print(f"Razones de verosimilitud: {ratios}")
    
    # Probabilidades posteriores
    prob_lesion, prob_non_lesion = classifier.posterior_probabilities(test_data)
    print(f"P(lesión|x): {prob_lesion}")
    print(f"P(no-lesión|x): {prob_non_lesion}")
    
    # Predicciones
    predictions = classifier.predict(test_data, threshold=1.0)
    print(f"Predicciones (umbral=1.0): {predictions}")
    
    # Parámetros del modelo
    params = classifier.get_model_parameters()
    print(f"\nParámetros del modelo:")
    print(f"Media lesión estimada: {params['mu_lesion']}")
    print(f"Media no-lesión estimada: {params['mu_non_lesion']}")