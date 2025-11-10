This repository contains the implementation of **W-MATRPO** (Wasserstein Multi-Agent Trust Region Policy Optimization) with **CAATR** (Coordination-Aware Adaptive Trust Region) mechanism, as described in the paper: Wasserstein-Constrained Trust Region Optimization for
Cooperative Multi-Agent Reinforcement Learning. 

All files are stand-alone, and should compile without any separate argumentation. Files represent the differential game environment, ablation study setup, and varying agent case. 


### Requirements
```bash
python >= 3.7
numpy >= 1.19.0
scipy >= 1.5.0
matplotlib >= 3.3.0
pandas >= 1.1.0
```

Note: Hyperparameters may need to be modified for correct experiment replication. 
### Hyperparameters

Key hyperparameters as specified in the paper (Table 2):

| Parameter | 3 Agents | 5 Agents | 7 Agents | 9 Agents |
|-----------|----------|----------|----------|----------|
| Iterations | 4000 | 4000 | 4000 | 4000 |
| Batch Size | 30 | 30 | 30 | 30 |
| CAATR Constant (C) | 0.02 | 0.02 | 0.10 | 0.15 |
| Trust Region (δ) | 0.1 | 0.1 | 0.1 | 0.1 |
| Initial μ₀ | 1.5 | 1.5 | 1.5 | 1.5 |
| Initial σ₀ | 0.5 | 0.5 | 0.5 | 0.5 |
| Critic LR | 0.2 | 0.2 | 0.2 | 0.2 |
