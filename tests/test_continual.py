import pytest
import numpy as np
from src.representation.autoencoder import CompositionalAutoencoder
from src.continual.continual_learner import ContinualLearner
from src.continual.metrics import ContinualLearningMetrics

def test_continual_learning_metrics():
    evaluator = ContinualLearningMetrics()
    matrix = [
        [0.85],
        [0.80, 0.88],
        [0.78, 0.85, 0.90]
    ]
    metrics = evaluator.log_evaluation_matrix(matrix)
    assert "average_accuracy" in metrics
    assert "backward_transfer" in metrics
    assert "catastrophic_forgetting_index" in metrics
    assert metrics["average_accuracy"] == round(np.mean([0.78, 0.85, 0.90]), 4)

def test_continual_learner_integration():
    model = CompositionalAutoencoder(input_dim=10, latent_dim=4)
    learner = ContinualLearner(model, replay_buffer_capacity=100)
    
    task1_data = np.random.randn(50, 10).astype(np.float32)
    task2_data = np.random.randn(50, 10).astype(np.float32)
    
    res1 = learner.learn_task(task1_data, task_id="task_1", epochs=2)
    assert "continual_metrics" in res1
    
    res2 = learner.learn_task(task2_data, task_id="task_2", epochs=2)
    assert len(res2["performance_row"]) == 2
