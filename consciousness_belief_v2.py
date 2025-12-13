#!/usr/bin/env python3
"""
Consciousness-Belief Propagation Simulation (Bayesian Version)
Based on Bala-Goyal social learning model

Key insight: Agents observe noisy evidence and update via Bayesian inference,
then share observations with neighbors. Network topology affects which evidence
propagates, leading to different convergence outcomes despite identical underlying
truth.

For consciousness beliefs:
- Action A = "Adopt consciousness-realism" (phenomenal consciousness is fundamental)
- Action B = "Adopt illusionism" (consciousness is narrative/functional)
- epsilon = probability that illusionism is actually better (set near 0.5 for ambiguity)
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import pandas as pd
from pathlib import Path
from scipy.special import expit  # sigmoid function


class BayesianBeliefNetwork:
    """
    Network of agents forming beliefs via Bayesian updating + social learning.

    Belief b_i ∈ [0, 1]: probability that "illusionism is better"
    - b_i = 0: Strong realist (phenomenal consciousness is fundamental)
    - b_i = 1: Strong illusionist (consciousness is narrative/functional)
    """

    def __init__(self, graph: nx.Graph, epsilon=0.51, trials_per_step=10,
                 initial_belief_range=(0.3, 0.7), seed=None):
        """
        Args:
            graph: Social network structure
            epsilon: True probability that illusionism is better (slight bias)
            trials_per_step: How many "experiments" each agent runs
            initial_belief_range: Initial belief distribution
            seed: Random seed for this network instance
        """
        self.graph = graph
        self.n_agents = graph.number_of_nodes()
        self.epsilon = epsilon  # True state of the world
        self.trials = trials_per_step

        # Set seed for this specific network instance
        if seed is not None:
            np.random.seed(seed)

        # Initialize beliefs uniformly at random
        self.beliefs = np.random.uniform(
            initial_belief_range[0],
            initial_belief_range[1],
            size=self.n_agents
        )

        # Track belief history
        self.history = [self.beliefs.copy()]

    def sample_evidence(self, node_id):
        """
        Agent conducts 'experiments' to test illusionism vs realism.

        Returns:
            (successes, trials): How many trials supported illusionism
        """
        # Only agents who currently believe illusionism (b > 0.5) run experiments
        if self.beliefs[node_id] <= 0.5:
            return 0, 0  # No evidence generated

        # Sample from binomial: each trial has epsilon probability of success
        successes = np.random.binomial(self.trials, self.epsilon)
        return successes, self.trials

    def bayesian_update(self, prior, successes, trials):
        """
        Bayesian update: posterior ∝ likelihood × prior

        Using beta-binomial conjugate prior for computational efficiency.
        Approximation: treat belief as beta distribution parameter.

        Args:
            prior: Prior belief that illusionism is better
            successes: Number of successful trials
            trials: Total trials

        Returns:
            Posterior belief
        """
        if trials == 0:
            return prior

        # Beta-binomial update (simplified)
        # posterior ∝ prior^successes × (1-prior)^(trials-successes)
        # This is approximate - proper Bayesian would track full beta distribution

        # Log-space to avoid numerical underflow
        log_prior = np.log(prior + 1e-10)  # Avoid log(0)
        log_likelihood = successes * log_prior + (trials - successes) * np.log(1 - prior + 1e-10)

        # Normalize (crude approximation)
        # Proper version would integrate over all possible epsilon values
        # Here we just do a weighted update
        alpha = 0.3  # Learning rate
        evidence_strength = successes / trials if trials > 0 else 0.5
        posterior = prior * (1 - alpha) + evidence_strength * alpha

        return np.clip(posterior, 0.0, 1.0)

    def step(self):
        """
        One step of social learning:
        1. Each agent samples evidence (if they believe illusionism)
        2. Agents share evidence with neighbors
        3. All agents update beliefs via Bayesian inference
        """
        # Store evidence for each agent
        evidence = {}
        for i in range(self.n_agents):
            evidence[i] = [self.sample_evidence(i)]  # Own evidence

        # Collect evidence from neighbors
        for i in range(self.n_agents):
            neighbors = list(self.graph.neighbors(i))
            for j in neighbors:
                if evidence[j][0][1] > 0:  # Neighbor has evidence
                    evidence[i].append(evidence[j][0])

        # Update beliefs based on all collected evidence
        new_beliefs = np.zeros(self.n_agents)
        for i in range(self.n_agents):
            belief = self.beliefs[i]

            # Aggregate all evidence
            total_successes = sum(succ for succ, trials in evidence[i])
            total_trials = sum(trials for succ, trials in evidence[i])

            # Bayesian update
            new_beliefs[i] = self.bayesian_update(belief, total_successes, total_trials)

        self.beliefs = new_beliefs
        self.history.append(self.beliefs.copy())

    def simulate(self, n_steps=1000, convergence_threshold=0.001, verbose=False):
        """
        Run simulation until convergence.

        Returns:
            Number of steps taken
        """
        for step in range(n_steps):
            old_beliefs = self.beliefs.copy()
            self.step()

            max_change = np.max(np.abs(self.beliefs - old_beliefs))

            if verbose and step % 100 == 0:
                print(f"Step {step}: mean={self.beliefs.mean():.4f}, "
                      f"std={self.beliefs.std():.4f}, max_change={max_change:.6f}")

            if max_change < convergence_threshold:
                if verbose:
                    print(f"Converged at step {step}")
                return step

        if verbose:
            print(f"Did not converge after {n_steps} steps")
        return n_steps

    def get_statistics(self) -> Dict[str, float]:
        """Return final belief statistics."""
        return {
            'mean': float(np.mean(self.beliefs)),
            'std': float(np.std(self.beliefs)),
            'min': float(np.min(self.beliefs)),
            'max': float(np.max(self.beliefs)),
            'median': float(np.median(self.beliefs)),
            'prop_illusionist': float(np.mean(self.beliefs > 0.5)),  # Fraction believing illusionism
            'prop_realist': float(np.mean(self.beliefs < 0.5))  # Fraction believing realism
        }


def create_networks(n=100) -> Dict[str, nx.Graph]:
    """Create test networks."""
    return {
        'complete': nx.complete_graph(n),
        'cycle': nx.cycle_graph(n),
        'small_world': nx.watts_strogatz_graph(n, 6, 0.3)  # Small-world (Facebook-like)
    }


def run_simulation_suite(n_agents=100, n_replications=10, n_steps=1000,
                          epsilon=0.51, trials_per_step=10, output_dir='results_v2'):
    """
    Run full simulation suite.

    Args:
        n_agents: Network size
        n_replications: Random seeds
        n_steps: Max simulation length
        epsilon: True probability illusionism is better (>0.5 = slight illusionist bias)
        trials_per_step: Evidence quality
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    networks = create_networks(n=n_agents)
    results = []

    print(f"Consciousness-Belief Propagation Simulation (Bayesian Version)")
    print(f"="*70)
    print(f"Parameters:")
    print(f"  - n_agents: {n_agents}")
    print(f"  - epsilon (true prob illusionism better): {epsilon}")
    print(f"  - trials_per_step: {trials_per_step}")
    print(f"  - n_replications: {n_replications}")
    print(f"\nExpected outcomes (Zollman effect):")
    print(f"  - Complete graph: Fast convergence, more likely to reach truth")
    print(f"  - Cycle graph: Slow convergence, may get stuck in false consensus")
    print(f"  - Small-world: Intermediate, possible polarization")
    print(f"="*70 + "\n")

    for topology_name, graph in networks.items():
        print(f"\n{'='*70}")
        print(f"Topology: {topology_name.upper()}")
        print(f"{'='*70}")

        for rep in range(n_replications):
            # Different seed for each replication
            seed = rep * 1000 + hash(topology_name) % 1000

            net = BayesianBeliefNetwork(
                graph,
                epsilon=epsilon,
                trials_per_step=trials_per_step,
                seed=seed
            )

            convergence_time = net.simulate(n_steps=n_steps, verbose=False)
            stats = net.get_statistics()

            result = {
                'topology': topology_name,
                'replication': rep,
                'seed': seed,
                'n_agents': n_agents,
                'epsilon': epsilon,
                'trials_per_step': trials_per_step,
                'convergence_time': convergence_time,
                **stats
            }
            results.append(result)

            if rep < 3:  # Print first few replications
                print(f"  Rep {rep+1}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, "
                      f"converged@{convergence_time}")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_path / 'simulation_results.csv', index=False)

    # Summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS (across replications)")
    print(f"{'='*70}\n")
    summary = df.groupby('topology').agg({
        'mean': ['mean', 'std'],
        'std': ['mean', 'std'],
        'prop_illusionist': ['mean', 'std'],
        'convergence_time': ['mean', 'std']
    }).round(4)
    print(summary)
    print(f"\nResults saved to {output_path / 'simulation_results.csv'}")

    return df


def plot_results(df: pd.DataFrame, output_dir='results_v2'):
    """Generate visualizations."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    topologies = sorted(df['topology'].unique())

    # Plot 1: Final mean belief
    ax = axes[0, 0]
    means = [df[df['topology'] == t]['mean'].values for t in topologies]
    ax.boxplot(means, tick_labels=topologies)
    ax.set_ylabel('Final Mean Belief')
    ax.set_title('Belief Convergence by Network Topology')
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Neutral')
    ax.axhline(y=df['epsilon'].iloc[0], color='g', linestyle='--', alpha=0.5,
               label=f'True value (ε={df["epsilon"].iloc[0]})')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 2: Belief disagreement
    ax = axes[0, 1]
    stds = [df[df['topology'] == t]['std'].values for t in topologies]
    ax.boxplot(stds, tick_labels=topologies)
    ax.set_ylabel('Final Std Dev (Disagreement)')
    ax.set_title('Belief Disagreement')
    ax.grid(alpha=0.3)

    # Plot 3: Proportion of illusionists
    ax = axes[1, 0]
    props = [df[df['topology'] == t]['prop_illusionist'].values for t in topologies]
    ax.boxplot(props, tick_labels=topologies)
    ax.set_ylabel('Proportion Believing Illusionism (b > 0.5)')
    ax.set_title('Belief Distribution')
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    ax.grid(alpha=0.3)

    # Plot 4: Convergence time
    ax = axes[1, 1]
    times = [df[df['topology'] == t]['convergence_time'].values for t in topologies]
    ax.boxplot(times, tick_labels=topologies)
    ax.set_ylabel('Convergence Time (steps)')
    ax.set_title('Convergence Speed')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'simulation_results.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path / 'simulation_results.png'}")
    plt.close()


def main():
    """Run simulations and generate results."""
    # Run with slight illusionist bias (epsilon > 0.5) to see if networks find truth
    df = run_simulation_suite(
        n_agents=100,
        n_replications=10,
        n_steps=1000,
        epsilon=0.51,  # Truth: illusionism is slightly better
        trials_per_step=10,
        output_dir='results_v2'
    )

    plot_results(df, output_dir='results_v2')

    print(f"\n{'='*70}")
    print("Simulation complete!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
