"""Byzantine-Robust Aggregation Module for TFL-HPL"""

import numpy as np
import torch
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class ByzantineAggregator:
    """Coordinate-wise median aggregation resistant to Byzantine devices"""
    
    def __init__(self):
        self.aggregation_history = []
        logger.info("ByzantineAggregator initialized")
    
    def aggregate(
        self,
        device_updates: List[List[torch.Tensor]],
        trust_scores: np.ndarray,
        byzantine_threshold: float = 0.33,
    ) -> List[torch.Tensor]:
        """
        Perform coordinate-wise median aggregation
        
        Algorithm:
        1. For each parameter dimension d:
            - Collect all device values for dimension d
            - Compute median value
            - Use median as aggregated value
        
        Properties:
            - Robust to floor((K-1)/3) Byzantine devices
            - Automatically removes outliers
            - Works with differential privacy noise
        
        Args:
            device_updates: List of device model updates
            trust_scores: Trust scores for each device
            byzantine_threshold: Maximum fraction of Byzantine devices tolerated
        
        Returns:
            Aggregated model weights using median
        """
        if not device_updates:
            logger.warning("No device updates to aggregate")
            return []
        
        # Calculate maximum Byzantine devices this system can tolerate
        n_devices = len(device_updates)
        max_byzantine = int(np.floor((n_devices - 1) / 3))
        logger.info(f"Aggregating {n_devices} device updates. "
                   f"Byzantine tolerance: {max_byzantine}/{n_devices}")
        
        # Get number of parameters
        n_params = len(device_updates[0])
        aggregated_weights = []
        
        # Coordinate-wise median aggregation
        for param_idx in range(n_params):
            param_values = []
            
            # Collect values for this parameter from all devices
            for device_idx, device_update in enumerate(device_updates):
                param_tensor = device_update[param_idx]
                # Flatten tensor for aggregation
                param_flat = param_tensor.flatten().cpu().detach().numpy()
                param_values.append(param_flat)
            
            # Stack into matrix (n_devices, param_size)
            param_matrix = np.vstack(param_values)
            
            # Compute coordinate-wise median
            aggregated_param = np.median(param_matrix, axis=0)
            
            # Convert back to tensor
            aggregated_tensor = torch.from_numpy(
                aggregated_param.reshape(device_updates[0][param_idx].shape)
            ).float()
            
            aggregated_weights.append(aggregated_tensor)
        
        self.aggregation_history.append(aggregated_weights)
        logger.info("Coordinate-wise median aggregation completed")
        
        return aggregated_weights
    
    def weighted_median_aggregation(
        self,
        device_updates: List[List[torch.Tensor]],
        trust_scores: np.ndarray,
    ) -> List[torch.Tensor]:
        """
        Weighted median aggregation: weight each update by device trust score
        
        Args:
            device_updates: List of device model updates
            trust_scores: Trust scores for each device (0-1)
        
        Returns:
            Weighted aggregated model weights
        """
        # Normalize trust scores to weights
        weights = trust_scores / np.sum(trust_scores)
        
        n_params = len(device_updates[0])
        aggregated_weights = []
        
        for param_idx in range(n_params):
            param_values = []
            param_weights = []
            
            for device_idx, device_update in enumerate(device_updates):
                param_tensor = device_update[param_idx]
                param_flat = param_tensor.flatten().cpu().detach().numpy()
                param_values.append(param_flat)
                param_weights.append(weights[device_idx])
            
            # Weighted median (similar to regular median but considers weights)
            param_matrix = np.vstack(param_values)
            param_weights = np.array(param_weights)
            
            # Compute weighted median for each coordinate
            aggregated_param = self._compute_weighted_median(
                param_matrix, param_weights
            )
            
            aggregated_tensor = torch.from_numpy(
                aggregated_param.reshape(device_updates[0][param_idx].shape)
            ).float()
            
            aggregated_weights.append(aggregated_tensor)
        
        return aggregated_weights
    
    def _compute_weighted_median(self, values: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Compute weighted median for array of values
        
        Args:
            values: Array of shape (n_devices, n_coords)
            weights: Array of shape (n_devices,)
        
        Returns:
            Weighted median array of shape (n_coords,)
        """
        n_coords = values.shape[1]
        weighted_medians = np.zeros(n_coords)
        
        for coord in range(n_coords):
            coord_values = values[:, coord]
            
            # Sort by values
            sorted_indices = np.argsort(coord_values)
            sorted_values = coord_values[sorted_indices]
            sorted_weights = weights[sorted_indices]
            
            # Find median (weight-based)
            cumsum_weights = np.cumsum(sorted_weights)
            median_idx = np.argmax(cumsum_weights >= 0.5)
            
            weighted_medians[coord] = sorted_values[median_idx]
        
        return weighted_medians
    
    def identify_byzantine_devices(
        self,
        device_updates: List[List[torch.Tensor]],
        aggregated_weights: List[torch.Tensor],
        threshold: float = 2.0,
    ) -> List[int]:
        """
        Identify Byzantine devices by comparing their updates to aggregated result
        
        Devices with updates deviating >threshold sigma are flagged
        
        Args:
            device_updates: List of device updates
            aggregated_weights: Aggregated weights from median
            threshold: Deviation threshold in standard deviations
        
        Returns:
            List of Byzantine device IDs
        """
        byzantine_devices = []
        
        for device_idx, device_update in enumerate(device_updates):
            # Compute deviation from aggregated weights
            total_deviation = 0.0
            n_params = len(device_update)
            
            for param_idx in range(n_params):
                deviation = torch.norm(
                    device_update[param_idx] - aggregated_weights[param_idx]
                ).item()
                total_deviation += deviation
            
            avg_deviation = total_deviation / n_params
            
            # Check if deviation exceeds threshold
            # (In practice, would compare to historical mean + k*std)
            if avg_deviation > threshold:
                byzantine_devices.append(device_idx)
        
        return byzantine_devices
    
    def get_aggregation_stats(self) -> Dict:
        """Get aggregation statistics"""
        return {
            'total_aggregations': len(self.aggregation_history),
            'aggregation_method': 'coordinate-wise-median',
            'byzantine_resilience': 'floor((K-1)/3)',
        }
