from typing import Dict, Any, List, Optional
import numpy as np
import torch

from src.representation.autoencoder import CompositionalAutoencoder
from src.continual.metrics import ContinualLearningMetrics

class ContinualLearner:
    """
    Continual Learning Coordinator.
    Sequentially adapts scientific representation models to new materials domains 
    using experience replay while safeguarding against catastrophic forgetting.
    """
    def __init__(
        self,
        representation_model: CompositionalAutoencoder,
        replay_buffer_capacity: int = 300,
        replay_sample_size: int = 32
    ):
        self.rep_model = representation_model
        self.replay_buffer_capacity = replay_buffer_capacity
        self.replay_sample_size = replay_sample_size
        self.replay_buffers: Dict[str, np.ndarray] = {}
        self.task_history: List[str] = []
        self.metrics_evaluator = ContinualLearningMetrics()
        self.performance_matrix: List[List[float]] = []

    def learn_task(
        self,
        new_task_data: np.ndarray,
        task_id: str,
        epochs: int = 8,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """Trains the representation model on the new task combined with replay memory."""
        self.task_history.append(task_id)
        
        # 1. Combine new data with existing replay samples
        combined_data = new_task_data
        if self.replay_buffers:
            replay_chunks = []
            for past_id, past_data in self.replay_buffers.items():
                n_sample = min(len(past_data), self.replay_sample_size)
                idx = np.random.choice(len(past_data), n_sample, replace=False)
                replay_chunks.append(past_data[idx])
            if replay_chunks:
                replay_combined = np.vstack(replay_chunks)
                combined_data = np.vstack([new_task_data, replay_combined])
                
        # 2. Train model
        train_res = self.rep_model.fit(combined_data, epochs=epochs, batch_size=batch_size)
        
        # 3. Update replay buffer for the current task
        n_store = min(len(new_task_data), self.replay_buffer_capacity // max(1, len(self.task_history)))
        sample_idx = np.random.choice(len(new_task_data), n_store, replace=False)
        self.replay_buffers[task_id] = new_task_data[sample_idx]
        
        # 4. Evaluate reconstruction loss on all tasks seen so far
        current_row = []
        for past_id in self.task_history:
            past_data = self.replay_buffers[past_id]
            rec_errors = self.rep_model.get_reconstruction_error(past_data)
            # Higher score = lower error (1 / (1 + MSE))
            score = float(1.0 / (1.0 + np.mean(rec_errors)))
            current_row.append(score)
            
        self.performance_matrix.append(current_row)
        cl_metrics = self.metrics_evaluator.log_evaluation_matrix(self.performance_matrix)
        
        return {
            "task_id": task_id,
            "final_train_loss": train_res["final_loss"],
            "performance_row": current_row,
            "continual_metrics": cl_metrics
        }
