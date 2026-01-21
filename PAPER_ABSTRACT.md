# TFL-HPL: IEEE GSEACT 2026 Research Paper

## Title

**Trustworthy Federated Learning with Heterogeneous Privacy Levels for Critical Infrastructure IoT: Byzantine-Resistant Distributed Learning with Device-Adaptive Privacy Budgets**

## Author

**Burra Deepak Yadav**

Department of Computer Science & Engineering
Madanapalle Institute of Technology and Science (MITS)
Bangarupalyem, Chittoor, Andhra Pradesh - 517112, India

📧 deepakyadavdeepu94@gmail.com

## Abstract

Critical infrastructure systems—smart grids, water distribution networks, and industrial control systems—face unprecedented cybersecurity threats while demanding privacy-preserving collaborative learning across heterogeneous, geographically distributed devices. Traditional federated learning frameworks assign uniform privacy budgets across all participants, failing to account for devices with varying security postures and trustworthiness levels.

This paper introduces **TFL-HPL (Trustworthy Federated Learning with Heterogeneous Privacy Levels)**, a novel Byzantine-resilient federated learning framework that assigns device-specific privacy budgets based on continuous trustworthiness scoring, enabling critical infrastructure networks to learn collaboratively while remaining resilient to compromised or malicious devices.

## Six Novel Contributions

### 1. Dynamic Trustworthiness Scoring ⭐
- Real-time reputation mechanism combining update consistency, anomaly detection, and historical reliability
- Markov Decision Process formalization with O(log(1/δ)) convergence guarantee
- First continuous trustworthiness spectrum in federated learning (not binary)

### 2. Personalized ε-Differential Privacy ⭐⭐
- **FIRST federated learning approach with device-level privacy budgets** (not global)
- Formula: ε_i = ε_global × (trust_score_i / Σtrust_scores)
- Formal proof: Personalized privacy preserves global (ε, δ)-differential privacy

### 3. Byzantine-Robust Aggregation with Privacy Preservation ⭐⭐
- Coordinate-wise median-based aggregation
- Proven resilience to ⌊(K-1)/3⌋ Byzantine devices while maintaining ε-DP
- **Novel finding**: Byzantine resilience IMPROVES when combined with differential privacy

### 4. Privacy Amplification During Attacks ⭐
- Detects Byzantine attacks through gradient divergence analysis
- Automatically amplifies privacy budgets for honest devices when attacks detected
- Implements "moving target defense" strategy

### 5. SCADA-Optimized Implementation ⭐
- Streaming differential privacy noise generation (O(1) memory instead of O(gradient_size))
- **First framework compatible with 256MB legacy SCADA controllers**
- Hardware validation on actual embedded systems (not simulation)

### 6. Comprehensive Critical Infrastructure Validation ⭐⭐
- Evaluated on IEEE 9-bus and 118-bus smart grid systems
- Water treatment facility dataset with SCADA-realistic constraints
- Hardware validation on heterogeneous devices (256MB SCADA to 8GB servers)
- Actual SCADA network topology (not simulated)

## Key Experimental Results

### Accuracy (IEEE 9-Bus Smart Grid)
- **94.7% accuracy** (vs. 88.7-91.3% for baselines)
- Maintains accuracy at strict privacy levels (ε=1.8)

### Convergence Speed
- **156% faster convergence** (243 rounds vs 520 for FedByzantine)
- Best privacy-utility trade-off across all baselines

### Byzantine Attack Detection
- **94.2-100% detection rate** across four adversarial scenarios
- <0.5% false positive rate
- Detects: label flipping, gradient inversion, poisoning, collusion

### Hardware Compatibility
- **Deployment feasibility** from 256MB SCADA to 8GB edge servers
- First framework achieving this hardware range

### Privacy Preservation
- **ε=1.8 privacy** with **85.4% utility**
- Satisfies HIPAA-equivalent privacy guarantees
- Exceeds NERC CIP, NIS Directive, CIIP compliance

## Comparison with State-of-the-Art

| Feature | Krum | Multi-Krum | FedByzantine | DP-FedAvg | **TFL-HPL** |
|---------|------|-----------|--------------|-----------|----------|
| Byzantine Resilience | ✓ | ✓ | ✓ | ~ | ✓ |
| Differential Privacy | ✗ | ✗ | ✓ | ✓ | ✓ |
| Personalized Privacy | ✗ | ✗ | ✗ | ✗ | ✓ **FIRST** |
| Trustworthiness Scoring | ✗ | ✗ | ✗ | ✗ | ✓ **FIRST** |
| Critical Infrastructure Focus | ✗ | ✗ | ✗ | ✗ | ✓ **FIRST** |
| 256MB SCADA Support | ✗ | ✗ | ✗ | ✗ | ✓ **FIRST** |
| Formal Proofs | ✓ | ✓ | ✓ | ✓ | ✓ |

## Paper Organization

- **Section 2**: Related work positioning TFL-HPL against 60+ papers
- **Sections 3-4**: Framework architecture and trustworthiness scoring
- **Sections 5-6**: Personalized privacy allocation and Byzantine-robust aggregation
- **Section 7**: Comprehensive experimental evaluation
- **Section 8**: Deployment considerations and regulatory compliance
- **Section 9**: Conclusions and future directions

## Submission Details

**Conference**: IEEE GSEACT 2026
**Status**: Ready for Submission
**Deadline**: February 15, 2026
**Category**: Security, Privacy, Byzantine Resilience
**Keywords**: Byzantine-Robust Federated Learning, Differential Privacy, Critical Infrastructure, Trustworthiness Scoring, Heterogeneous IoT, SCADA Systems, Smart Grids, Adaptive Privacy Budgets, Cyber-Physical Systems

## Implementation

Complete open-source implementation available at:
**https://github.com/deepakdeepu-12/TFL-HPL**

## Citation

```bibtex
@inproceedings{yadav2026tfl,
  title={Trustworthy Federated Learning with Heterogeneous Privacy Levels for Critical Infrastructure IoT},
  author={Yadav, Burra Deepak},
  booktitle={IEEE GSEACT 2026},
  year={2026}
}
```

---

**Ready for Critical Infrastructure Deployment** 🚀
