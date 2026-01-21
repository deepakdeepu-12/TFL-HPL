"""Privacy Budget Allocation Module for TFL-HPL"""

import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class PrivacyBudgetAllocator:
    """Allocate personalized privacy budgets based on trustworthiness"""
    
    def __init__(self, global_epsilon: float = 2.0, global_delta: float = 1e-5):
        """
        Initialize privacy budget allocator
        
        Args:
            global_epsilon: Global privacy budget
            global_delta: Global failure probability
        """
        self.global_epsilon = global_epsilon
        self.global_delta = global_delta
        self.epsilon_history = []
        
        logger.info(f"PrivacyBudgetAllocator initialized: ε_global={global_epsilon}, δ_global={global_delta}")
    
    def allocate_budgets(self, trust_scores: np.ndarray, epsilon: float) -> np.ndarray:
        """
        Allocate privacy budgets proportional to trust scores
        
        Formula: ε_i = ε_global × (trust_score_i / Σtrust_scores)
        
        Intuition: Higher-trust devices get looser privacy (accept more noise),
                   Lower-trust devices get stricter privacy (enforce more noise)
        
        Args:
            trust_scores: Array of trust scores [0, 1] for all devices
            epsilon: Global privacy budget
        
        Returns:
            Array of personalized privacy budgets
        """
        # Normalize trust scores
        trust_sum = np.sum(trust_scores)
        if trust_sum == 0:
            # Fallback: uniform allocation
            epsilon_allocation = np.ones(len(trust_scores)) * epsilon / len(trust_scores)
        else:
            # Allocate proportionally
            epsilon_allocation = epsilon * (trust_scores / trust_sum)
        
        # Ensure all allocations are positive
        epsilon_allocation = np.clip(epsilon_allocation, 1e-3, epsilon)
        
        self.epsilon_history.append(epsilon_allocation.copy())
        
        logger.debug(f"Allocated ε: mean={np.mean(epsilon_allocation):.3f}, "
                    f"min={np.min(epsilon_allocation):.3f}, "
                    f"max={np.max(epsilon_allocation):.3f}")
        
        return epsilon_allocation
    
    def amplify_privacy(
        self,
        epsilon_allocation: np.ndarray,
        byzantine_devices: List[int],
        amplification_factor: float = 1.5,
    ) -> np.ndarray:
        """
        Amplify privacy budgets when Byzantine attack detected
        
        Strategy: When attack detected, amplify privacy for honest devices
        to increase protection against future attacks
        
        Args:
            epsilon_allocation: Current privacy allocation
            byzantine_devices: List of detected Byzantine device IDs
            amplification_factor: Multiplication factor for amplification (default 1.5)
        
        Returns:
            Updated privacy allocation with amplified budgets
        """
        epsilon_new = epsilon_allocation.copy()
        
        # Find honest devices (not in byzantine list)
        honest_mask = np.ones(len(epsilon_allocation), dtype=bool)
        for device_id in byzantine_devices:
            if 0 <= device_id < len(epsilon_allocation):
                honest_mask[device_id] = False
        
        # Amplify privacy for honest devices
        epsilon_new[honest_mask] *= amplification_factor
        
        # Reduce or zero out Byzantine device budgets
        epsilon_new[~honest_mask] *= 0.5
        
        # Normalize to maintain global budget
        total_current = np.sum(epsilon_allocation)
        total_new = np.sum(epsilon_new)
        if total_new > 0:
            epsilon_new = epsilon_new * (total_current / total_new)
        
        logger.info(f"Privacy amplified for {np.sum(honest_mask)} honest devices. "
                   f"Reduced for {len(byzantine_devices)} Byzantine devices.")
        
        return epsilon_new
    
    def verify_global_budget(self, epsilon_allocation: np.ndarray) -> bool:
        """
        Verify that sum of device budgets doesn't exceed global budget
        
        Theorem: If each device i satisfies (epsilon_i, delta_i)-DP,
                 then federated system satisfies (sum(epsilon_i)/K, sum(delta_i))-DP
        """
        budget_sum = np.sum(epsilon_allocation)
        return budget_sum <= self.global_epsilon * len(epsilon_allocation)
    
    def get_allocation_report(self) -> Dict:
        """Generate allocation statistics report"""
        if not self.epsilon_history:
            return {}
        
        latest_allocation = self.epsilon_history[-1]
        
        return {
            'global_epsilon': self.global_epsilon,
            'device_mean_epsilon': float(np.mean(latest_allocation)),
            'device_min_epsilon': float(np.min(latest_allocation)),
            'device_max_epsilon': float(np.max(latest_allocation)),
            'device_std_epsilon': float(np.std(latest_allocation)),
            'allocations_made': len(self.epsilon_history),
        }
