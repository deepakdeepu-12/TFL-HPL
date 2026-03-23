# TFL-HPL: Trustworthy Federated Learning with Heterogeneous Privacy Levels

**Byzantine-Resistant Distributed Learning for Critical Infrastructure IoT**

![Status](https://img.shields.io/badge/status-production-brightgreen)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![IEEE](https://img.shields.io/badge/IEEE-GSEACT%202026-red)

## Overview

TFL-HPL is a federated learning framework specifically designed for critical infrastructure systems (smart grids, water treatment, industrial control). It combines:

- **Byzantine-Robust Aggregation**: Resilient to >33% compromised devices
- **Personalized Privacy Budgets**: Device-adaptive ε-differential privacy allocation
- **Dynamic Trustworthiness Scoring**: Real-time reputation mechanism
- **Privacy Amplification**: Automatic privacy boost during attacks
- **Legacy Hardware Support**: Runs on 256MB SCADA controllers

## Key Results

- ✅ **94.7% accuracy** on IEEE 9-bus smart grid (ε=1.8 privacy)
- ✅ **156% faster convergence** than existing Byzantine-robust methods
- ✅ **94.2-100% attack detection** accuracy
- ✅ **Hardware compatibility**: 256MB SCADA → 8GB servers

## Quick Installation

```bash
git clone https://github.com/deepakdeepu-12/TFL-HPL.git
cd TFL-HPL
pip install -r requirements.txt
```

## Quick Start

```python
from tfl_hpl import TFLCoordinator, TFLDevice
from datasets import load_ieee_9bus

# Load critical infrastructure dataset
train_data, test_data = load_ieee_9bus()

# Initialize coordinator
coordinator = TFLCoordinator(
    n_devices=50,
    global_epsilon=2.0,
    byzantine_threshold=0.33,
    rounds=100
)

# Add devices with different trust levels
for i in range(50):
    device = TFLDevice(
        device_id=i,
        data=train_data[i],
        device_type='scada' if i < 10 else 'iot',
        memory_mb=256 if i < 10 else 1024
    )
    coordinator.add_device(device)

# Run federated training
history = coordinator.train()

# Evaluate on test set
accuracy, privacy_budget = coordinator.evaluate(test_data)
print(f"Accuracy: {accuracy:.2%}, Privacy: ε={privacy_budget:.2f}")
```

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│  Global Coordinator (Edge Gateway / Cloud)                 │
│  • Trustworthiness Scoring                                 │
│  • Privacy Budget Allocation                               │
│  • Byzantine-Robust Aggregation                            │
└────────────────────────────────────────────────────────────┘
         ↑            ↑            ↑
    TLS 1.3      TLS 1.3      TLS 1.3
         ↓            ↓            ↓
    ┌─────────┐  ┌─────────┐  ┌─────────┐
    │ SCADA   │  │ IoT     │  │ Edge    │
    │Trust:   │  │Trust:   │  │Trust:   │
    │0.95     │  │0.70     │  │0.35     │
    │ε=0.927  │  │ε=0.732  │  │ε=0.341  │
    └─────────┘  └─────────┘  └─────────┘
```

## Core Components

1. **Trustworthiness Scoring** - Markov chain-based trust model
2. **Privacy Budget Allocation** - Personalized ε-differential privacy
3. **Byzantine Aggregation** - Coordinate-wise median aggregation
4. **Attack Detection** - 94.2-100% detection rate

## Datasets

- IEEE 9-Bus (50 sensors)
- IEEE 118-Bus (200 sensors)
- Water Treatment Facility (45 sensors, 2 years)

## Performance

| Metric | Result | vs. Baseline |
|--------|--------|----------|
| Accuracy (IEEE 9-bus) | 94.7% | +3.4% |
| Privacy (ε) | 1.8 | -64% |
| Convergence | 243 rounds | 156% faster |
| Attack Detection | 94.2% | First |
| Hardware Support | 256MB SCADA | **First framework** |

## Publication

**Conference**: IEEE GSEACT 2026  
**Status**: Submission completed 
**Deadline**: February 15, 2026  

## Citation

```bibtex
@inproceedings{yadav2026tfl,
  title={Trustworthy Federated Learning with Heterogeneous Privacy Levels for Critical Infrastructure IoT},
  author={Yadav, Burra Deepak},
  booktitle={IEEE GSEACT 2026},
  year={2026}
}
```

## License

MIT License - See LICENSE file

## Contact

📧 Email: deepakyadavdeepu94@gmail.com

---

**Ready for Critical Infrastructure Deployment** 🚀
