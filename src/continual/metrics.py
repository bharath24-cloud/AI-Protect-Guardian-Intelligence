from typing import Dict, Any, List
import numpy as np

class ContinualLearningMetrics:
    """
    Computes rigorous metrics for sequential scientific representation learning:
    - Matrix R[j, i]: Performance on task i after training on task j (j >= i)
    - Average Retention / Accuracy
    - Backward Transfer (BWT)
    - Catastrophic Forgetting Index
    """
    def __init__(self):
        self.performance_matrix: List[List[float]] = []

    def log_evaluation_matrix(self, matrix: List[List[float]]) -> Dict[str, float]:
        """Calculates continual learning benchmarks from an upper-triangular or full performance matrix."""
        self.performance_matrix = matrix
        K = len(matrix)
        if K <= 1:
            return {
                "average_accuracy": matrix[0][0] if matrix else 0.0,
                "backward_transfer": 0.0,
                "catastrophic_forgetting_index": 0.0
            }
            
        # Final row R[K-1, i]
        final_scores = [matrix[K-1][i] for i in range(K)]
        avg_acc = float(np.mean(final_scores))
        
        # Backward Transfer (BWT) = (1 / (K-1)) * sum_{i=0}^{K-2} (R[K-1, i] - R[i, i])
        bwt_terms = [matrix[K-1][i] - matrix[i][i] for i in range(K - 1)]
        bwt = float(np.mean(bwt_terms))
        
        # Forgetting F_K = (1 / (K-1)) * sum_{i=0}^{K-2} (max_{l < K-1} R[l, i] - R[K-1, i])
        forgetting_terms = []
        for i in range(K - 1):
            peak_prior = max(matrix[l][i] for l in range(i, K - 1))
            forgetting_terms.append(max(0.0, peak_prior - matrix[K-1][i]))
            
        forgetting_index = float(np.mean(forgetting_terms))
        
        return {
            "average_accuracy": round(avg_acc, 4),
            "backward_transfer": round(bwt, 4),
            "catastrophic_forgetting_index": round(forgetting_index, 4)
        }
