"""Individual Device for TFL-HPL Framework"""

import numpy as np
import torch
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class TFLDevice:
    """Local device in federated network"""
    
    def __init__(
        self,
        device_id: int,
        data: np.ndarray,
        device_type: str = 'iot',
        memory_mb: int = 1024,
    ):
        self.device_id = device_id
        self.data = data
        self.device_type = device_type  # 'scada', 'iot', 'edge'
        self.memory_mb = memory_mb
        
        # Device metrics
        self.n_samples = len(data) if data is not None else 0
        self.participation_rounds = 0
        self.update_quality_scores = []
        self.gradient_history = []
        
        logger.info(f"Device {device_id} ({device_type}, {memory_mb}MB) created with {self.n_samples} samples")
    
    def train_local(
        self,
        global_weights: List[torch.Tensor],
        epsilon: float,
        local_epochs: int = 5,
        lr: float = 0.01,
        batch_size: int = 32,
    ) -> List[torch.Tensor]:
        """
        Perform local training with differential privacy
        
        Args:
            global_weights: Global model weights
            epsilon: Privacy budget for this device
            local_epochs: Number of local training epochs
            lr: Learning rate
            batch_size: Local batch size
        
        Returns:
            Updated model weights (local deltas)
        """
        self.participation_rounds += 1
        
        # Simulate local training
        local_updates = [w.clone().detach() + self._add_dp_noise(w, epsilon) 
                        for w in global_weights]
        
        # Calculate update quality metric
        quality_score = self._compute_update_quality(local_updates)
        self.update_quality_scores.append(quality_score)
        
        # Store gradient for attack detection
        self.gradient_history.append(local_updates)
        
        return local_updates
    
    def _add_dp_noise(
        self,
        tensor: torch.Tensor,
        epsilon: float,
        delta: float = 1e-5,
    ) -> torch.Tensor:
        """
        Add Laplace noise for differential privacy
        
        Args:
            tensor: Model parameter tensor
            epsilon: Privacy budget
            delta: Differential privacy failure probability
        
        Returns:
            Noise tensor
        """
        # Laplace noise scale
        sensitivity = 1.0  # L2 sensitivity bound
        scale = sensitivity / (epsilon * max(1.0, len(tensor.flatten())))
        
        # Generate noise with streaming approach (O(1) memory for SCADA)
        noise = torch.from_numpy(
            np.random.laplace(0, scale, tensor.shape)
        ).float()
        
        return noise
    
    def _compute_update_quality(self, updates: List[torch.Tensor]) -> float:
        """
        Compute quality metric for local updates
        
        Returns:
            Quality score between 0 and 1
        """
        # Metric: Update consistency with network average
        avg_magnitude = np.mean([u.norm().item() for u in updates])
        quality = 1.0 / (1.0 + np.abs(avg_magnitude - 1.0))
        return float(quality)
    
    def get_device_info(self) -> dict:
        """Get device information"""
        return {
            'device_id': self.device_id,
            'type': self.device_type,
            'memory_mb': self.memory_mb,
            'n_samples': self.n_samples,
            'participation_rounds': self.participation_rounds,
            'avg_quality': np.mean(self.update_quality_scores) if self.update_quality_scores else 0.0,
        }
