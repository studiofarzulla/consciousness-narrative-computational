#!/usr/bin/env python3
"""
Consciousness-Belief Propagation Simulation
Standalone implementation for consciousness-as-narrative paper

Demonstrates that consciousness-beliefs stabilize based on network structure
rather than metaphysical truth, using social learning dynamics.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import pandas as pd
from pathlib import Path


# Evidence weights (sum to zero - no net evidence)
W_INTRO = -0.30  # Introspective evidence (supports realism)
W_NEURO = +0.20  # Neuroscientific evidence (supports illusionism)
W_PHIL = +0.10   # Philosophical evidence (supports illusionism)
E_TOTAL = W_INTRO + W_NEURO + W_PHIL  # Should be 0.0

# Belief update parameters
ALPHA = 0.60  # Conservatism weight (stick to current belief)
BETA = 0.30   # Social influence weight (neighbors' average)
GAMMA = 0.10  # Evidence weight (evidence signal)


class ConsciousnessBeliefNetwork:
    """
    Network of agents forming beliefs about consciousness.

    Belief scale:
      0.0 = Complete consciousness-realism (phenomenal consciousness is ontological primitive)
      0.5 = Uncertain/intermediate
      1.0 = Complete illusionism (consciousness is narrative/functional)
    """

    def __init__(self, graph: nx.Graph, initial_belief_range=(0.3, 0.7)):
        """
        Args:
            graph: NetworkX graph representing social network
            initial_belief_range: (min, max) for uniform random initialization
        """
        self.graph = graph
        self.n_agents = graph.number_of_nodes()

        # Initialize beliefs uniformly at random
        np.random.seed(42)  # For reproducibility
        self.beliefs = np.random.uniform(
            initial_belief_range[0],
            initial_belief_range[1],
            size=self.n_agents
        )

        # Store belief history
        self.history = [self.beliefs.copy()]

    def update_beliefs(self, alpha=ALPHA, beta=BETA, gamma=GAMMA, evidence=E_TOTAL):
        """
        Update all agents' beliefs based on:
        1. Their current belief (conservatism)
        2. Their neighbors' beliefs (social influence)
        3. Evidence signal (rational updating)

        b_i(t+1) = α·b_i(t) + β·mean(neighbors) + γ·E_total
        """
        new_beliefs = np.zeros(self.n_agents)

        for i in range(self.n_agents):
            # Current belief
            current = self.beliefs[i]

            # Neighbors' average belief
            neighbors = list(self.graph.neighbors(i))
            if neighbors:
                neighbor_avg = np.mean([self.beliefs[j] for j in neighbors])
            else:
                neighbor_avg = current  # Isolated node stays unchanged

            # Update rule
            new_belief = alpha * current + beta * neighbor_avg + gamma * evidence

            # Clip to [0, 1]
            new_beliefs[i] = np.clip(new_belief, 0.0, 1.0)

        self.beliefs = new_beliefs
        self.history.append(self.beliefs.copy())

    def simulate(self, n_steps=1000, convergence_threshold=0.001, verbose=True):
        """
        Run simulation until convergence or max steps.

        Returns:
            n_steps_taken: Number of steps until convergence
        """
        for step in range(n_steps):
            old_beliefs = self.beliefs.copy()
            self.update_beliefs()

            # Check convergence (max absolute change < threshold)
            max_change = np.max(np.abs(self.beliefs - old_beliefs))

            if verbose and step % 100 == 0:
                print(f"Step {step}: mean={self.beliefs.mean():.4f}, "
                      f"std={self.beliefs.std():.4f}, max_change={max_change:.6f}")

            if max_change < convergence_threshold:
                if verbose:
                    print(f"\nConverged at step {step}")
                return step

        if verbose:
            print(f"\nDid not converge after {n_steps} steps")
        return n_steps

    def get_statistics(self) -> Dict[str, float]:
        """Return final belief statistics."""
        return {
            'mean': float(np.mean(self.beliefs)),
            'std': float(np.std(self.beliefs)),
            'min': float(np.min(self.beliefs)),
            'max': float(np.max(self.beliefs)),
            'median': float(np.median(self.beliefs))
        }


def create_networks(n=100) -> Dict[str, nx.Graph]:
    """
    Create the three test networks.

    Args:
        n: Number of nodes

    Returns:
        Dictionary of network name -> networkx graph
    """
    return {
        'complete': nx.complete_graph(n),
        'cycle': nx.cycle_graph(n),
        # For now, use a simple random graph as Facebook proxy
        # TODO: Load actual Facebook ego network dataset
        'facebook': nx.watts_strogatz_graph(n, 10, 0.1)  # Small-world network
    }


def run_simulation_suite(n_agents=100, n_replications=10, n_steps=1000,
                          output_dir='results'):
    """
    Run full simulation suite across all network topologies.

    Args:
        n_agents: Number of agents per network
        n_replications: Number of random initializations
        n_steps: Max simulation steps
        output_dir: Directory to save results
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    networks = create_networks(n=n_agents)
    results = []

    print(f"Running {n_replications} replications for each of {len(networks)} topologies...")
    print(f"Parameters: α={ALPHA}, β={BETA}, γ={GAMMA}, E_total={E_TOTAL}\n")

    for topology_name, graph in networks.items():
        print(f"\n{'='*60}")
        print(f"Topology: {topology_name.upper()}")
        print(f"{'='*60}")

        for rep in range(n_replications):
            print(f"\nReplication {rep+1}/{n_replications}")

            # Create network and simulate
            net = ConsciousnessBeliefNetwork(graph)
            convergence_time = net.simulate(n_steps=n_steps, verbose=False)
            stats = net.get_statistics()

            # Store results
            result = {
                'topology': topology_name,
                'replication': rep,
                'n_agents': n_agents,
                'convergence_time': convergence_time,
                **stats
            }
            results.append(result)

            print(f"  Converged at step {convergence_time}")
            print(f"  Final: mean={stats['mean']:.4f}, std={stats['std']:.4f}")

    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(output_path / 'simulation_results.csv', index=False)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}\n")
    summary = df.groupby('topology').agg({
        'mean': ['mean', 'std'],
        'std': ['mean', 'std'],
        'convergence_time': ['mean', 'std']
    })
    print(summary)

    return df


def plot_results(df: pd.DataFrame, output_dir='results'):
    """
    Generate visualizations of simulation results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    topologies = df['topology'].unique()

    # Plot 1: Final mean belief by topology
    ax = axes[0]
    means = df.groupby('topology')['mean'].apply(list)
    ax.boxplot([means[t] for t in topologies], labels=topologies)
    ax.set_ylabel('Final Mean Belief')
    ax.set_title('Belief Convergence by Network Topology')
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Neutral (0.5)')
    ax.legend()

    # Plot 2: Final standard deviation (disagreement)
    ax = axes[1]
    stds = df.groupby('topology')['std'].apply(list)
    ax.boxplot([stds[t] for t in topologies], labels=topologies)
    ax.set_ylabel('Final Std Dev (Disagreement)')
    ax.set_title('Belief Disagreement by Topology')

    # Plot 3: Convergence time
    ax = axes[2]
    times = df.groupby('topology')['convergence_time'].apply(list)
    ax.boxplot([times[t] for t in topologies], labels=topologies)
    ax.set_ylabel('Convergence Time (steps)')
    ax.set_title('Convergence Speed by Topology')

    plt.tight_layout()
    plt.savefig(output_path / 'simulation_results.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_path / 'simulation_results.png'}")
    plt.close()


def main():
    """Run the full simulation and generate results."""
    print("Consciousness-Belief Propagation Simulation")
    print("=" * 60)
    print("\nResearch question: Do consciousness-beliefs stabilize based on")
    print("network structure rather than metaphysical truth?")
    print("\nHypothesis (Zollman effect):")
    print("  - Complete graph → rapid convergence to realism (b → 0)")
    print("  - Cycle graph → slow convergence to illusionism (b → 1)")
    print("  - Small-world → persistent disagreement (multimodal)")
    print("\n" + "="*60 + "\n")

    # Run simulations
    df = run_simulation_suite(
        n_agents=100,
        n_replications=10,
        n_steps=1000,
        output_dir='results'
    )

    # Generate plots
    plot_results(df, output_dir='results')

    print("\n" + "="*60)
    print("Simulation complete! Results saved to ./results/")
    print("="*60)


if __name__ == '__main__':
    main()
