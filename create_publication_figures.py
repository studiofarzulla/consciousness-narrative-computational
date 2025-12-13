"""
Publication-quality figures for PolyGraphs network epistemology results.

Creates 300 DPI figures demonstrating Zollman effect for consciousness-beliefs:
- Complete graph: Fast convergence to truth
- Cycle graph: Slow convergence to FALSE consensus
- Small-world: Persistent disagreement (realistic)

Author: Studio Farzulla Research
Date: November 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Publication settings
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
})

# Color-blind friendly palette (Tol palette)
COLORS = {
    'complete': '#4477AA',    # Blue
    'cycle': '#EE6677',       # Red/Pink
    'small_world': '#228833', # Green
}

# Truth value and neutral line
EPSILON = 0.51
NEUTRAL = 0.5

# Load data
DATA_DIR = Path('/home/kawaiikali/Documents/Resurrexi/projects/needs-work/consciousness-narrative-paper/results_v2')
OUTPUT_DIR = DATA_DIR / 'figures_publication'
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_DIR / 'simulation_results.csv')

# Rename topologies for publication
topology_labels = {
    'complete': 'Complete',
    'cycle': 'Cycle',
    'small_world': 'Small-World'
}
df['topology_label'] = df['topology'].map(topology_labels)


def figure1_belief_convergence():
    """Figure 1: Belief Convergence by Topology (Boxplot)"""
    fig, ax = plt.subplots(figsize=(6, 4.5))

    # Boxplot
    box_parts = ax.boxplot(
        [df[df['topology'] == 'complete']['mean'],
         df[df['topology'] == 'cycle']['mean'],
         df[df['topology'] == 'small_world']['mean']],
        labels=['Complete', 'Cycle', 'Small-World'],
        patch_artist=True,
        widths=0.6,
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor='black', markersize=6)
    )

    # Color boxes
    for patch, topology in zip(box_parts['boxes'], ['complete', 'cycle', 'small_world']):
        patch.set_facecolor(COLORS[topology])
        patch.set_alpha(0.7)

    # Reference lines
    ax.axhline(NEUTRAL, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Neutral (0.5)')
    ax.axhline(EPSILON, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Truth (ε = {EPSILON})')

    # Annotate false consensus
    cycle_mean = df[df['topology'] == 'cycle']['mean'].mean()
    ax.annotate(
        f'False Consensus\n({cycle_mean:.3f})',
        xy=(2, cycle_mean),
        xytext=(2.5, cycle_mean - 0.02),
        fontsize=9,
        color=COLORS['cycle'],
        weight='bold',
        arrowprops=dict(arrowstyle='->', color=COLORS['cycle'], lw=1.5)
    )

    ax.set_ylabel('Final Mean Belief', fontsize=11)
    ax.set_ylim(0.42, 0.53)
    ax.set_title('Network Topology Determines Belief Convergence', fontsize=12, pad=15)
    ax.legend(loc='upper right', frameon=True, edgecolor='gray')
    ax.grid(axis='y', alpha=0.3, linestyle=':')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure1_belief_convergence.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Figure 1 saved: {OUTPUT_DIR / 'figure1_belief_convergence.png'}")


def figure2_speed_accuracy_tradeoff():
    """Figure 2: Convergence Speed vs Truth-Tracking (Scatter)"""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    # Calculate distance from truth
    df['distance_from_truth'] = np.abs(df['mean'] - EPSILON)

    # Scatter plot for each topology
    for topology in ['complete', 'cycle', 'small_world']:
        data = df[df['topology'] == topology]
        ax.scatter(
            data['convergence_time'],
            data['distance_from_truth'],
            c=COLORS[topology],
            label=topology_labels[topology],
            s=100,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )

    # Annotate key findings
    cycle_data = df[df['topology'] == 'cycle']
    complete_data = df[df['topology'] == 'complete']

    # Cycle: slow + wrong
    cycle_x = cycle_data['convergence_time'].mean()
    cycle_y = cycle_data['distance_from_truth'].mean()
    ax.annotate(
        'Slow convergence\nto FALSEHOOD',
        xy=(cycle_x, cycle_y),
        xytext=(cycle_x + 100, cycle_y + 0.01),
        fontsize=8,
        color=COLORS['cycle'],
        weight='bold',
        arrowprops=dict(arrowstyle='->', color=COLORS['cycle'], lw=1.2)
    )

    # Complete: fast + correct
    complete_x = complete_data['convergence_time'].mean()
    complete_y = complete_data['distance_from_truth'].mean()
    ax.annotate(
        'Fast convergence\nto truth',
        xy=(complete_x, complete_y),
        xytext=(complete_x + 150, complete_y - 0.005),
        fontsize=8,
        color=COLORS['complete'],
        weight='bold',
        arrowprops=dict(arrowstyle='->', color=COLORS['complete'], lw=1.2)
    )

    ax.set_xlabel('Convergence Time (steps)', fontsize=11)
    ax.set_ylabel('Distance from Truth |mean - 0.51|', fontsize=11)
    ax.set_title('Zollman Effect: Speed-Accuracy Trade-off', fontsize=12, pad=15)
    ax.legend(loc='upper right', frameon=True, edgecolor='gray')
    ax.grid(alpha=0.3, linestyle=':')
    ax.set_xlim(0, 1050)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure2_speed_accuracy_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Figure 2 saved: {OUTPUT_DIR / 'figure2_speed_accuracy_tradeoff.png'}")


def figure3_persistent_disagreement():
    """Figure 3: Persistent Disagreement (Standard Deviation)"""
    fig, ax = plt.subplots(figsize=(6, 4.5))

    # Boxplot of disagreement
    box_parts = ax.boxplot(
        [df[df['topology'] == 'complete']['std'],
         df[df['topology'] == 'cycle']['std'],
         df[df['topology'] == 'small_world']['std']],
        labels=['Complete', 'Cycle', 'Small-World'],
        patch_artist=True,
        widths=0.6,
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor='black', markersize=6)
    )

    # Color boxes
    for patch, topology in zip(box_parts['boxes'], ['complete', 'cycle', 'small_world']):
        patch.set_facecolor(COLORS[topology])
        patch.set_alpha(0.7)

    # Annotate small-world persistence
    sw_mean = df[df['topology'] == 'small_world']['std'].mean()
    ax.annotate(
        'Persistent\nDisagreement',
        xy=(3, sw_mean),
        xytext=(2.4, sw_mean + 0.004),
        fontsize=9,
        color=COLORS['small_world'],
        weight='bold',
        arrowprops=dict(arrowstyle='->', color=COLORS['small_world'], lw=1.5)
    )

    ax.set_ylabel('Belief Disagreement (std)', fontsize=11)
    ax.set_title('Realistic Networks Maintain Persistent Disagreement', fontsize=12, pad=15)
    ax.grid(axis='y', alpha=0.3, linestyle=':')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure3_persistent_disagreement.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Figure 3 saved: {OUTPUT_DIR / 'figure3_persistent_disagreement.png'}")


def figure4_four_panel_summary():
    """Figure 4: Four-Panel Summary (2x2 grid)"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top-left: Mean belief
    ax = axes[0, 0]
    box_parts = ax.boxplot(
        [df[df['topology'] == 'complete']['mean'],
         df[df['topology'] == 'cycle']['mean'],
         df[df['topology'] == 'small_world']['mean']],
        labels=['Complete', 'Cycle', 'Small-World'],
        patch_artist=True,
        widths=0.6
    )
    for patch, topology in zip(box_parts['boxes'], ['complete', 'cycle', 'small_world']):
        patch.set_facecolor(COLORS[topology])
        patch.set_alpha(0.7)
    ax.axhline(NEUTRAL, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(EPSILON, color='green', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_ylabel('Final Mean Belief')
    ax.set_title('(A) Belief Convergence', fontsize=11, weight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle=':')

    # Top-right: Convergence time
    ax = axes[0, 1]
    box_parts = ax.boxplot(
        [df[df['topology'] == 'complete']['convergence_time'],
         df[df['topology'] == 'cycle']['convergence_time'],
         df[df['topology'] == 'small_world']['convergence_time']],
        labels=['Complete', 'Cycle', 'Small-World'],
        patch_artist=True,
        widths=0.6
    )
    for patch, topology in zip(box_parts['boxes'], ['complete', 'cycle', 'small_world']):
        patch.set_facecolor(COLORS[topology])
        patch.set_alpha(0.7)
    ax.set_ylabel('Convergence Time (steps)')
    ax.set_title('(B) Convergence Speed', fontsize=11, weight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle=':')
    ax.set_yscale('log')

    # Bottom-left: Disagreement (std)
    ax = axes[1, 0]
    box_parts = ax.boxplot(
        [df[df['topology'] == 'complete']['std'],
         df[df['topology'] == 'cycle']['std'],
         df[df['topology'] == 'small_world']['std']],
        labels=['Complete', 'Cycle', 'Small-World'],
        patch_artist=True,
        widths=0.6
    )
    for patch, topology in zip(box_parts['boxes'], ['complete', 'cycle', 'small_world']):
        patch.set_facecolor(COLORS[topology])
        patch.set_alpha(0.7)
    ax.set_ylabel('Belief Disagreement (std)')
    ax.set_title('(C) Persistent Disagreement', fontsize=11, weight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle=':')

    # Bottom-right: Proportion illusionist
    ax = axes[1, 1]
    box_parts = ax.boxplot(
        [df[df['topology'] == 'complete']['prop_illusionist'],
         df[df['topology'] == 'cycle']['prop_illusionist'],
         df[df['topology'] == 'small_world']['prop_illusionist']],
        labels=['Complete', 'Cycle', 'Small-World'],
        patch_artist=True,
        widths=0.6
    )
    for patch, topology in zip(box_parts['boxes'], ['complete', 'cycle', 'small_world']):
        patch.set_facecolor(COLORS[topology])
        patch.set_alpha(0.7)
    ax.axhline(0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='50% threshold')
    ax.set_ylabel('Proportion Illusionist')
    ax.set_title('(D) Final Belief Distribution', fontsize=11, weight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle=':')
    ax.set_ylim(-0.05, 1.05)

    fig.suptitle('Network Epistemology of Consciousness-Beliefs', fontsize=14, weight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure4_four_panel_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Figure 4 saved: {OUTPUT_DIR / 'figure4_four_panel_summary.png'}")


def generate_summary_stats():
    """Generate summary statistics for manuscript"""
    stats = {}

    for topology in ['complete', 'cycle', 'small_world']:
        data = df[df['topology'] == topology]
        stats[topology] = {
            'mean_belief': f"{data['mean'].mean():.4f} ± {data['mean'].std():.4f}",
            'mean_convergence': f"{data['convergence_time'].mean():.1f} ± {data['convergence_time'].std():.1f}",
            'mean_disagreement': f"{data['std'].mean():.4f} ± {data['std'].std():.4f}",
            'distance_from_truth': f"{np.abs(data['mean'] - EPSILON).mean():.4f}",
        }

    # Save to text file
    with open(OUTPUT_DIR / 'summary_statistics.txt', 'w') as f:
        f.write("PolyGraphs Network Epistemology - Summary Statistics\n")
        f.write("=" * 60 + "\n\n")

        for topology in ['complete', 'cycle', 'small_world']:
            f.write(f"{topology.upper()}\n")
            f.write("-" * 40 + "\n")
            for key, value in stats[topology].items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

        f.write("\nKEY FINDINGS:\n")
        f.write("-" * 40 + "\n")

        cycle_mean = df[df['topology'] == 'cycle']['mean'].mean()
        f.write(f"1. Cycle graph FALSE CONSENSUS: {cycle_mean:.4f} (truth = {EPSILON})\n")
        f.write(f"   Distance from truth: {abs(cycle_mean - EPSILON):.4f}\n\n")

        complete_time = df[df['topology'] == 'complete']['convergence_time'].mean()
        cycle_time = df[df['topology'] == 'cycle']['convergence_time'].mean()
        f.write(f"2. Convergence speed: Complete ({complete_time:.1f} steps) vs Cycle ({cycle_time:.1f} steps)\n")
        f.write(f"   Cycle is {cycle_time/complete_time:.1f}x slower\n\n")

        sw_std = df[df['topology'] == 'small_world']['std'].mean()
        complete_std = df[df['topology'] == 'complete']['std'].mean()
        f.write(f"3. Small-world disagreement: {sw_std:.4f} vs Complete: {complete_std:.4f}\n")
        f.write(f"   Small-world has {sw_std/complete_std:.0f}x more disagreement\n")

    print(f"✓ Summary statistics saved: {OUTPUT_DIR / 'summary_statistics.txt'}")


if __name__ == '__main__':
    print("Generating publication-quality figures (300 DPI)...\n")

    figure1_belief_convergence()
    figure2_speed_accuracy_tradeoff()
    figure3_persistent_disagreement()
    figure4_four_panel_summary()
    generate_summary_stats()

    print(f"\n✓ All figures saved to: {OUTPUT_DIR}")
    print("\nFigures generated:")
    print("  - figure1_belief_convergence.png")
    print("  - figure2_speed_accuracy_tradeoff.png")
    print("  - figure3_persistent_disagreement.png")
    print("  - figure4_four_panel_summary.png")
    print("  - summary_statistics.txt")
