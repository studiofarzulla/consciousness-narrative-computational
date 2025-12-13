# Consciousness Narrative Computational Models

Computational validation code for **"Replication Optimization at Scale: Dissolving Qualia via Occam's Razor"** (Farzulla, 2025).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17917970.svg)](https://doi.org/10.5281/zenodo.17917970)

## Overview

This repository contains network epistemology simulations validating predictions from the consciousness-as-narrative thesis. The core finding: **network structure, not philosophical depth, explains why consciousness debates persist**.

Built on the [PolyGraphs](https://github.com/alexandroskoliousis/polygraphs) framework (Koliousis, 2024).

## Key Results

| Network Topology | Convergence | Final Belief | Interpretation |
|------------------|-------------|--------------|----------------|
| Complete Graph | 100% | 0.30 (truth) | Full connectivity → truth wins |
| Cycle Graph | 100% | 1.0 (realist) | Echo chambers → systematic error |
| Small-World | 0% | 0.51 ± 0.035 | Realistic structure → persistent disagreement |

## Quick Start

```bash
# Clone
git clone https://github.com/studiofarzulla/consciousness-narrative-computational.git
cd consciousness-narrative-computational

# Install
pip install -r requirements.txt

# Run simulations
python consciousness_belief_v2.py --topologies all --seeds 10

# Generate publication figures
python create_publication_figures.py --output publication_figures/
```

## Repository Structure

```
├── consciousness_belief_sim.py    # Basic simulation script
├── consciousness_belief_v2.py     # Enhanced simulation with all topologies
├── create_publication_figures.py  # Figure generation for paper
├── results/                       # Simulation output data
│   └── simulation_results.csv
├── publication_figures/           # Generated figures
├── polygraphs/                    # PolyGraphs framework (core library)
├── configs/                       # Simulation configurations
├── examples/                      # Usage examples
└── scripts/                       # Utility scripts
```

## Simulation Parameters

- **Agents**: 100 per simulation
- **Topologies**: Complete, Cycle, Small-World (Watts-Strogatz k=4, p=0.1)
- **Truth value**: 0.3 (illusionism favored)
- **Update rule**: Bayesian belief revision
- **Convergence**: < 0.01 belief variance or 1000 steps
- **Replications**: 10 per condition

## Requirements

- Python 3.10+
- NumPy, Pandas, Matplotlib
- NetworkX

See `requirements.txt` for full dependencies.

## Citation

```bibtex
@misc{farzulla2025consciousness,
  author = {Farzulla, Murad},
  title = {Replication Optimization at Scale: Dissolving Qualia via Occam's Razor},
  year = {2025},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.17917970}
}
```

## Acknowledgments

Built on the [PolyGraphs](https://github.com/alexandroskoliousis/polygraphs) framework:

> Ball, B., Koliousis, A., Mohanan, A. & Peacey, M. [Computational philosophy: reflections on the PolyGraphs project](https://doi.org/10.1057/s41599-024-02619-z). Humanit Soc Sci Commun 11, 186 (2024).

## License

MIT License - see [LICENSE](LICENSE)

## Author

**Murad Farzulla** - [Farzulla Research](https://farzulla.org)
ORCID: [0009-0002-7164-8704](https://orcid.org/0009-0002-7164-8704)
