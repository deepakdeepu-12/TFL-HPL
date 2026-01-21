"""Attack Detection Module for TFL-HPL"""

import numpy as np
import torch
from typing import List, Dict, Tuple
from sklearn.ensemble import IsolationForest
import logging

logger = logging.getLogger(__name__)


class AttackDetector:
    """Detect Byzantine attacks using multiple methods"""
    
    ATTACK_TYPES = {
        'label_flipping': 'Flip gradient magnitude',
        'gradient_inversion': 'Flip gradient signs',
        'model_poisoning': 'Subtle systematic shift',
        'collusion': 'Coordinated attack',
    }
    
    def __init__(self):
        self.detection_history = []
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        logger.info("AttackDetector initialized")
    
    def detect_attacks(
        self,
        device_updates: List[List[torch.Tensor]],
        trust_scores: np.ndarray,
        detection_threshold: float = 0.5,
    ) -> List[int]:
        """
        Detect Byzantine attacks using multiple methods
        
        Methods:
        1. Gradient magnitude analysis
        2. Consistency checking
        3. Anomaly detection (Isolation Forest)
        4. Statistical outlier detection
        
        Args:
            device_updates: List of device model updates
            trust_scores: Trust scores for each device
            detection_threshold: Confidence threshold for detection
        
        Returns:
            List of detected Byzantine device IDs
        """
        byzantine_devices = []
        
        # Method 1: Gradient magnitude analysis
        magnitude_detected = self._detect_by_magnitude(device_updates, trust_scores)
        
        # Method 2: Consistency checking
        consistency_detected = self._detect_by_consistency(device_updates, trust_scores)
        
        # Method 3: Anomaly detection
        anomaly_detected = self._detect_by_anomaly(device_updates)
        
        # Combine results: device is Byzantine if detected by >1 method
        all_detections = magnitude_detected + consistency_detected + anomaly_detected
        detection_counts = {}
        for device_id in all_detections:
            detection_counts[device_id] = detection_counts.get(device_id, 0) + 1
        
        # Flag devices detected by multiple methods
        for device_id, count in detection_counts.items():
            if count >= 2:  # Require 2+ detection methods to flag
                byzantine_devices.append(device_id)
        
        if byzantine_devices:
            logger.warning(f"Byzantine attacks detected from devices: {byzantine_devices}")
        
        self.detection_history.append({
            'timestamp': len(self.detection_history),
            'detected_devices': byzantine_devices,
            'magnitude_detected': magnitude_detected,
            'consistency_detected': consistency_detected,
            'anomaly_detected': anomaly_detected,
        })
        
        return byzantine_devices
    
    def _detect_by_magnitude(self, device_updates: List, trust_scores: np.ndarray) -> List[int]:
        """
        Detect Byzantine by unusual gradient magnitude
        
        Byzantine attacks often produce very large or very small gradients
        """
        magnitudes = []
        
        for device_update in device_updates:
            total_magnitude = 0.0
            for param in device_update:
                magnitude = torch.norm(param).item()
                total_magnitude += magnitude
            magnitudes.append(total_magnitude)
        
        magnitudes = np.array(magnitudes)
        
        # Statistical outlier detection: >2 sigma from mean
        mean_mag = np.mean(magnitudes)
        std_mag = np.std(magnitudes)
        
        detected = []
        for device_id, mag in enumerate(magnitudes):
            z_score = np.abs((mag - mean_mag) / (std_mag + 1e-6))
            if z_score > 2.0:  # 2 sigma threshold
                detected.append(device_id)
        
        return detected
    
    def _detect_by_consistency(self, device_updates: List, trust_scores: np.ndarray) -> List[int]:
        """
        Detect Byzantine by low consistency with peers
        
        Honest devices typically have consistent updates,
        Byzantine devices diverge from consensus
        """
        detected = []
        n_devices = len(device_updates)
        
        # Compute pairwise similarity (simplified)
        for device_id in range(n_devices):
            similarities = []
            
            for other_id in range(n_devices):
                if device_id == other_id:
                    continue
                
                # Compute cosine similarity
                similarity = self._compute_update_similarity(
                    device_updates[device_id],
                    device_updates[other_id]
                )
                similarities.append(similarity)
            
            # Low average similarity indicates potential Byzantine
            avg_similarity = np.mean(similarities) if similarities else 1.0
            if avg_similarity < 0.6:  # Low consistency threshold
                detected.append(device_id)
        
        return detected
    
    def _detect_by_anomaly(self, device_updates: List) -> List[int]:
        """
        Detect Byzantine using Isolation Forest anomaly detection
        """
        # Convert updates to feature vectors
        feature_vectors = []
        for device_update in device_updates:
            features = []
            for param in device_update:
                flat = param.flatten().cpu().detach().numpy()
                # Extract statistics
                features.extend([
                    np.mean(flat),
                    np.std(flat),
                    np.max(np.abs(flat)),
                ])
            feature_vectors.append(features)
        
        if len(feature_vectors) < 2:
            return []
        
        feature_array = np.array(feature_vectors)
        
        # Isolation Forest detection
        try:
            predictions = self.isolation_forest.fit_predict(feature_array)
            detected = [i for i, pred in enumerate(predictions) if pred == -1]
        except:
            detected = []
        
        return detected
    
    def _compute_update_similarity(self, update1: List, update2: List) -> float:
        """
        Compute similarity between two model updates
        Using cosine similarity
        """
        # Flatten updates
        flat1 = torch.cat([p.flatten() for p in update1])
        flat2 = torch.cat([p.flatten() for p in update2])
        
        # Cosine similarity
        similarity = torch.nn.functional.cosine_similarity(
            flat1.unsqueeze(0),
            flat2.unsqueeze(0)
        ).item()
        
        return similarity
    
    def classify_attack_type(
        self,
        device_id: int,
        device_update: List[torch.Tensor],
        aggregated_weights: List[torch.Tensor],
    ) -> str:
        """
        Classify type of attack if Byzantine detected
        
        Returns:
            Attack type string
        """
        # Compute deviation characteristics
        param_deviations = []
        for i, param in enumerate(device_update):
            deviation = torch.norm(param - aggregated_weights[i]).item()
            param_deviations.append(deviation)
        
        max_deviation = max(param_deviations)
        avg_deviation = np.mean(param_deviations)
        
        # Simple heuristic classification
        if max_deviation > 5.0:
            return 'label_flipping'
        elif avg_deviation > 2.0:
            return 'gradient_inversion'
        else:
            return 'model_poisoning'
    
    def get_detection_report(self) -> Dict:
        """Get attack detection statistics"""
        total_rounds = len(self.detection_history)
        rounds_with_attacks = sum(1 for h in self.detection_history if h['detected_devices'])
        total_detected = sum(len(h['detected_devices']) for h in self.detection_history)
        
        return {
            'total_rounds': total_rounds,
            'rounds_with_attacks': rounds_with_attacks,
            'total_devices_detected': total_detected,
            'detection_rate': rounds_with_attacks / total_rounds if total_rounds > 0 else 0.0,
            'avg_devices_per_attack': total_detected / rounds_with_attacks if rounds_with_attacks > 0 else 0.0,
        }
