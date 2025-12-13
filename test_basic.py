#!/usr/bin/env python3
"""Test basic PolyGraphs functionality"""

import torch
import numpy as np
import polygraphs as pg
from polygraphs import hyperparameters as hparams
from polygraphs import ops

# Create a simple configuration
params = hparams.PolyGraphHyperParameters()
params.init.kind = 'uniform'
params.epsilon = 0.01
params.network.kind = 'complete'
params.network.size = 10
params.logging.enabled = True
params.logging.interval = 50
params.simulation.steps = 100
params.simulation.repeats = 1
params.seed = 123456789

# pg.random(params.seed)  # Skip - old DGL doesn't have dgl.random module
torch.manual_seed(params.seed)
np.random.seed(params.seed)
result = pg.simulate(params, op=ops.BalaGoyalOp)

print("\nSimulation completed successfully!")
print(f"Result type: {type(result)}")
