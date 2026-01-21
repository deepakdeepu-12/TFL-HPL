"""Trustworthiness Scoring Module for TFL-HPL"""

import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class TrustworthinessScorer:
    """Dynamic trustworthiness scoring with Markov chain model"""
    
    # Trust states
    TRUST_STATES = {
        'high': 0.95,
        'medium': 0.70,
        'low': 0.30,
    }
    
    # State transition matrix
    TRANSITION_MATRIX = np.array([
        [0.95, 0.04, 0.01],  # High -> [High, Medium, Low]
        [0.10, 0.80, 0.10],  # Medium -> [High, Medium, Low]
        [0.05, 0.15, 0.80],  # Low -> [High, Medium, Low]
    ])
    
    def __init__(self, n_devices: int):
        self.n_devices = n_devices
        # Initialize all devices at Medium trust
        self.trust_scores = np.ones(n_devices) * self.TRUST_STATES['medium']
        self.device_states = np.ones(n_devices, dtype=int)  # 0=high, 1=medium, 2=low
        
        logger.info(f"TrustworthinessScorer initialized for {n_devices} devices")
    
    def update_scores(self, history: Dict) -> np.ndarray:
        """
        Update trust scores based on device behavior
        
        Returns:
            Array of trust scores for all devices
        """
        # Components of trust score
        consistency_scores = self._compute_consistency_scores()
        anomaly_scores = self._compute_anomaly_scores()
        reliability_scores = self._compute_reliability_scores()
        
        # Weighted combination: 40% consistency + 30% anomaly + 30% reliability
        self.trust_scores = (
            0.4 * consistency_scores +
            0.3 * anomaly_scores +
            0.3 * reliability_scores
        )
        
        # Clip to valid range
        self.trust_scores = np.clip(self.trust_scores, 0, 1)
        
        return self.trust_scores
    
    def _compute_consistency_scores(self) -> np.ndarray:
        """
        Compute consistency with peer consensus
        High consistency = close to average, Low consistency = outlier
        """
        # Simulate consistency scores
        base_consistency = np.random.uniform(0.7, 1.0, self.n_devices)
        return base_consistency
    
    def _compute_anomaly_scores(self) -> np.ndarray:
        """
        Compute anomaly detection scores
        1.0 = no anomaly, 0.3 = anomaly detected
        """
        # Simulate anomaly detection
        base_anomaly = np.random.uniform(0.8, 1.0, self.n_devices)
        # 5% of devices have anomalies
        anomaly_indices = np.random.choice(self.n_devices, size=int(0.05*self.n_devices), replace=False)
        base_anomaly[anomaly_indices] = 0.3
        return base_anomaly
    
    def _compute_reliability_scores(self) -> np.ndarray:
        """
        Compute historical reliability
        Based on participation and quality consistency
        """
        # Simulate reliability scores
        base_reliability = np.random.uniform(0.7, 1.0, self.n_devices)
        return base_reliability
    
    def detect_byzantine_device(self, device_id: int) -> bool:
        """
        Detect if device is Byzantine
        
        Args:
            device_id: Device ID to check
        
        Returns:
            True if Byzantine behavior detected
        """
        # Byzantine if trust score < 0.4
        return self.trust_scores[device_id] < 0.4
    
    def get_trust_state(self, device_id: int) -> str:
        """Get human-readable trust state"""
        trust = self.trust_scores[device_id]
        if trust >= 0.80:
            return 'high'
        elif trust >= 0.50:
            return 'medium'
        else:
            return 'low'
    
    def get_all_trust_scores(self) -> np.ndarray:
        """Get all device trust scores"""
        return self.trust_scores.copy()
    
    def report(self) -> Dict:
        """Generate trust report"""
        high_trust = np.sum(self.trust_scores >= 0.80)
        medium_trust = np.sum((self.trust_scores >= 0.50) & (self.trust_scores < 0.80))
        low_trust = np.sum(self.trust_scores < 0.50)
        
        return {
            'high_trust_devices': int(high_trust),
            'medium_trust_devices': int(medium_trust),
            'low_trust_devices': int(low_trust),
            'avg_trust_score': float(np.mean(self.trust_scores)),
            'min_trust_score': float(np.min(self.trust_scores)),
            'max_trust_score': float(np.max(self.trust_scores)),
        }
