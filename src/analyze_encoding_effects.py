# ------------------------------------------------------------------------------------------
# Description: Analyze encoder/fusion network effects on sample separability
# ------------------------------------------------------------------------------------------

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from loguru import logger

# Setup paths
current_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(current_dir)
MODEL_DIR = os.path.join(current_dir, "model")
OUTPUT_DIR = os.path.join(current_dir, "output")
CONFIG_DIR = os.path.join(current_dir, "config")
DATA_DIR = os.path.join(current_dir, "data")

from src.model_pinn import PINN


def compute_pairwise_distances(features, metric='euclidean'):
    """
    Compute pairwise distances between all samples.

    Args:
        features: numpy array of shape (n_samples, n_features)
        metric: 'euclidean' or 'cosine'

    Returns:
        distances: 1D array of all pairwise distances (upper triangle)
    """
    n_samples = features.shape[0]

    if metric == 'euclidean':
        # Compute pairwise Euclidean distances
        distances = []
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = np.linalg.norm(features[i] - features[j])
                distances.append(dist)
    elif metric == 'cosine':
        # Compute pairwise cosine distances (1 - cosine_similarity)
        # Normalize features first
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        normalized = features / norms

        distances = []
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                cosine_sim = np.dot(normalized[i], normalized[j])
                cosine_dist = 1 - cosine_sim
                distances.append(cosine_dist)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return np.array(distances)


def extract_representations(model, input_features, c_scaling_factor):
    """
    Extract intermediate representations from the model.

    Args:
        model: PINN model
        input_features: Input tensor [batch_size, n_features]
        c_scaling_factor: Scaling factor for concentrations

    Returns:
        dict with raw_features, sample_encoding, time_encoding, fused_features
    """
    model.eval()
    device = model.device

    with torch.no_grad():
        # Move input to device
        input_features = input_features.to(device)

        # Preprocess inputs (same as in forward pass)
        c_initial, env_dict = model.preprocess_initial_inputs(x=input_features)

        # Replace C* concentrations with XF* concentrations (as done in training)
        c_initial = torch.cat([
            c_initial[:, :3],
            input_features[:, 5:13] * c_scaling_factor,
            c_initial[:, 11:]
        ], dim=1)

        batch_size = c_initial.shape[0]

        # Create normalized temporal context
        max_target_time = c_initial[:, 0].max()
        epsilon = model.epsilon
        normalized_target_time = c_initial[:, 0] / (max_target_time + epsilon)
        temporal_context = normalized_target_time.unsqueeze(1)

        # Extract sample-specific features (same as in forward)
        sample_features = torch.cat([
            c_initial[:, 1:2],   # T
            c_initial[:, 2:3],   # pH
            c_initial[:, 3:11],  # initial_conc
            c_initial[:, 11:14], # P, FEED, FH2
            temporal_context,    # Normalized target time
        ], dim=1)

        # Get sample encoding
        sample_encoding = model.sample_encoder(sample_features)

        # Use actual target times from data (normalized)
        # This better reflects how time encoding varies across samples
        t_batch_tensor = normalized_target_time.unsqueeze(1)

        # Get time encoding (now varies per sample based on their actual target time)
        time_encoding = model.time_encoder(t_batch_tensor)

        # Get fused features
        fused_features = model.feature_fusion(time_encoding, sample_encoding)

        return {
            'raw_features': sample_features.cpu().numpy(),
            'sample_encoding': sample_encoding.cpu().numpy(),
            'time_encoding': time_encoding.cpu().numpy(),
            'fused_features': fused_features.cpu().numpy()
        }


def plot_distance_distributions(distances_dict, output_dir, metric='euclidean'):
    """
    Plot histograms of pairwise distances for different representations.

    Args:
        distances_dict: dict mapping representation name to distances array
        output_dir: Directory to save plots
        metric: Distance metric used
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

    for idx, (name, distances) in enumerate(distances_dict.items()):
        ax = axes[idx]

        # Plot histogram
        n, bins, patches = ax.hist(distances, bins=50, density=True,
                                    alpha=0.7, color=colors[idx], edgecolor='black')

        # Fit and plot normal distribution
        mu, std = stats.norm.fit(distances)
        x = np.linspace(distances.min(), distances.max(), 100)
        pdf = stats.norm.pdf(x, mu, std)
        ax.plot(x, pdf, 'k--', linewidth=2, label=f'Normal fit\n$\mu$={mu:.3f}, $\sigma$={std:.3f}')

        # Add statistics text box
        cv = std / mu if mu != 0 else 0  # Coefficient of variation
        stats_text = f'Mean: {np.mean(distances):.4f}\nStd: {np.std(distances):.4f}\nCV: {cv:.4f}'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel(f'{metric.capitalize()} Distance', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'{name}', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Pairwise Distance Distributions ({metric.capitalize()})',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f'distance_distributions_{metric}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved distance distribution plot to {save_path}")


def plot_distance_overlay(distances_dict, output_dir, metric='euclidean'):
    """
    Plot overlaid histograms comparing raw input and sample encoding distributions.
    This clearly shows the expansion of the distribution after encoding.

    Args:
        distances_dict: dict mapping representation name to distances array
        output_dir: Directory to save plots
        metric: Distance metric used
    """
    # Figure size with 1.5:1 aspect ratio (width:height)
    fig, ax = plt.subplots(figsize=(7.5, 5))

    raw_distances = distances_dict['Raw Input (Standardized)']
    sample_distances = distances_dict['Sample Encoding']

    # Determine common bin edges for fair comparison
    all_distances = np.concatenate([raw_distances, sample_distances])
    bins = np.linspace(all_distances.min(), all_distances.max(), 50)

    # Plot raw input distribution
    ax.hist(raw_distances, bins=bins, density=True, alpha=0.6,
            color='#2E86AB', edgecolor='#1a5276', linewidth=1.2,
            label=f'Original input\n$\mu$={np.mean(raw_distances):.3f}, $\sigma$={np.std(raw_distances):.3f}')

    # Plot sample encoding distribution
    ax.hist(sample_distances, bins=bins, density=True, alpha=0.6,
            color='#E74C3C', edgecolor='#922b21', linewidth=1.2,
            label=f'Sample encoding\n$\mu$={np.mean(sample_distances):.3f}, $\sigma$={np.std(sample_distances):.3f}')

    # Add vertical lines for means
    raw_mean = np.mean(raw_distances)
    sample_mean = np.mean(sample_distances)
    ax.axvline(raw_mean, color='#1a5276', linestyle='--',
               linewidth=2, alpha=0.8)
    ax.axvline(sample_mean, color='#922b21', linestyle='--',
               linewidth=2, alpha=0.8)

    # Add arrow showing the shift between means
    y_pos = ax.get_ylim()[1] * 0.85
    ax.annotate('', xy=(sample_mean, y_pos), xytext=(raw_mean, y_pos),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2.5))
    ax.text((raw_mean + sample_mean) / 2, y_pos * 1.05,
            f'+{((sample_mean - raw_mean) / raw_mean * 100):.1f}%',
            ha='center', va='bottom', fontsize=14, fontweight='bold', color='#2c3e50')

    # Metric label with only first word capitalized
    metric_label = metric.capitalize() + ' distance'
    ax.set_xlabel(metric_label, fontsize=16)
    ax.set_ylabel('Density', fontsize=16)
    ax.legend(loc='center right', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=14)

    # Add text box with improvement summary (upper right corner)
    improvement = (sample_mean - raw_mean) / raw_mean * 100
    cv_raw = np.std(raw_distances) / np.mean(raw_distances)
    cv_sample = np.std(sample_distances) / np.mean(sample_distances)
    summary_text = (f'Mean distance increase: {improvement:.1f}%\n'
                    f'CV: {cv_raw:.3f} → {cv_sample:.3f}')
    ax.text(0.98, 0.98, summary_text, transform=ax.transAxes, fontsize=13,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor='gray'))

    plt.tight_layout()

    save_path = os.path.join(output_dir, f'distance_overlay_{metric}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved distance overlay plot to {save_path}")


def plot_distance_comparison(distances_dict, output_dir, metric='euclidean'):
    """
    Plot box plots comparing distance distributions across representations.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    data = [distances_dict[name] for name in distances_dict.keys()]
    labels = list(distances_dict.keys())

    bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=True)

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel(f'{metric.capitalize()} Distance', fontsize=12)
    ax.set_xlabel('Representation', fontsize=12)
    ax.set_title(f'Distance Distribution Comparison ({metric.capitalize()})',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add mean markers
    means = [np.mean(d) for d in data]
    ax.scatter(range(1, len(means) + 1), means, marker='D', color='red',
               s=100, zorder=3, label='Mean')
    ax.legend()

    plt.tight_layout()

    save_path = os.path.join(output_dir, f'distance_comparison_{metric}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved distance comparison plot to {save_path}")


def plot_tsne_visualization(representations_dict, output_dir, labels=None):
    """
    Create t-SNE visualizations of different representation spaces.

    Args:
        representations_dict: dict mapping representation name to features array
        output_dir: Directory to save plots
        labels: Optional labels for coloring points (e.g., temperature or pressure)
    """
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

        # Improved t-SNE settings
        default_perplexity = 20  # Good balance for ~100 samples
        n_iter = 1500  # Increased for better convergence
        init_method = 'pca'  # More stable and reproducible

        for idx, (name, features) in enumerate(representations_dict.items()):
            ax = axes[idx]

            try:
                # Skip if features have too few dimensions
                if features.shape[1] < 2:
                    ax.text(0.5, 0.5, 'Not enough dimensions\nfor t-SNE',
                            ha='center', va='center', fontsize=12)
                    ax.set_title(f'{name}', fontsize=14, fontweight='bold')
                    continue

                # Standardize features before t-SNE
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features)

                # Adjust perplexity based on sample size
                perplexity = min(default_perplexity, max(5, features.shape[0] // 4))
                if perplexity < 5 or features.shape[0] < 10:
                    ax.text(0.5, 0.5, 'Not enough samples\nfor t-SNE',
                            ha='center', va='center', fontsize=12)
                    ax.set_title(f'{name}', fontsize=14, fontweight='bold')
                    continue

                tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                            n_iter=n_iter, init=init_method, learning_rate='auto')
                features_2d = tsne.fit_transform(features_scaled)

                # Plot
                if labels is not None:
                    scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1],
                                        c=labels, cmap='viridis', alpha=0.7, s=50)
                    plt.colorbar(scatter, ax=ax, label='Temperature (K)')
                else:
                    ax.scatter(features_2d[:, 0], features_2d[:, 1],
                              alpha=0.7, s=50, c='#2E86AB')

                ax.set_xlabel('t-SNE 1', fontsize=12)
                ax.set_ylabel('t-SNE 2', fontsize=12)
                ax.set_title(f'{name}\n(perplexity={perplexity}, iter={n_iter})',
                             fontsize=13, fontweight='bold')
                ax.grid(True, alpha=0.3)

            except Exception as e:
                logger.warning(f"t-SNE failed for {name}: {e}")
                ax.text(0.5, 0.5, f't-SNE failed:\n{str(e)[:30]}',
                        ha='center', va='center', fontsize=10)
                ax.set_title(f'{name}', fontsize=14, fontweight='bold')

        plt.suptitle('t-SNE Visualization of Representation Spaces\n(PCA init, auto learning rate)',
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        save_path = os.path.join(output_dir, 'tsne_visualization.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved t-SNE visualization to {save_path}")

    except Exception as e:
        logger.error(f"Failed to create t-SNE visualization: {e}")
        plt.close('all')


def plot_tsne_perplexity_comparison(representations_dict, output_dir, labels=None):
    """
    Create t-SNE visualizations with different perplexity values to help choose optimal settings.
    Focuses on Raw Input vs Sample Encoding comparison.

    Args:
        representations_dict: dict mapping representation name to features array
        output_dir: Directory to save plots
        labels: Optional labels for coloring points
    """
    perplexity_values = [10, 20, 30]
    n_iter = 1500
    init_method = 'pca'

    # Compare Raw Input and Sample Encoding
    compare_names = ['Raw Input (Standardized)', 'Sample Encoding']

    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        for row_idx, name in enumerate(compare_names):
            if name not in representations_dict:
                continue

            features = representations_dict[name]

            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            for col_idx, perplexity in enumerate(perplexity_values):
                ax = axes[row_idx, col_idx]

                try:
                    # Adjust perplexity if needed
                    actual_perplexity = min(perplexity, features.shape[0] - 1)
                    if actual_perplexity < 5:
                        ax.text(0.5, 0.5, 'Not enough samples',
                                ha='center', va='center', fontsize=12)
                        ax.set_title(f'{name}\nperplexity={perplexity}')
                        continue

                    tsne = TSNE(n_components=2, perplexity=actual_perplexity, random_state=42,
                                n_iter=n_iter, init=init_method, learning_rate='auto')
                    features_2d = tsne.fit_transform(features_scaled)

                    # Plot
                    if labels is not None:
                        scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1],
                                            c=labels, cmap='viridis', alpha=0.7, s=50)
                        if col_idx == 2:  # Add colorbar only to rightmost column
                            plt.colorbar(scatter, ax=ax, label='Temperature (K)')
                    else:
                        ax.scatter(features_2d[:, 0], features_2d[:, 1],
                                  alpha=0.7, s=50, c='#2E86AB')

                    ax.set_xlabel('t-SNE 1', fontsize=11)
                    ax.set_ylabel('t-SNE 2', fontsize=11)
                    ax.set_title(f'{name}\nperplexity={actual_perplexity}',
                                 fontsize=12, fontweight='bold')
                    ax.grid(True, alpha=0.3)

                except Exception as e:
                    logger.warning(f"t-SNE failed for {name} with perplexity={perplexity}: {e}")
                    ax.text(0.5, 0.5, f't-SNE failed:\n{str(e)[:30]}',
                            ha='center', va='center', fontsize=10)
                    ax.set_title(f'{name}\nperplexity={perplexity}')

        plt.suptitle('t-SNE Perplexity Comparison: Raw Input vs Sample Encoding\n'
                     f'(PCA init, {n_iter} iterations, auto learning rate)',
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        save_path = os.path.join(output_dir, 'tsne_perplexity_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved t-SNE perplexity comparison to {save_path}")

    except Exception as e:
        logger.error(f"Failed to create t-SNE perplexity comparison: {e}")
        plt.close('all')


def plot_tsne_raw_vs_encoding(representations_dict, output_dir, temp_labels, pressure_labels, xfa_labels=None):
    """
    Create a 3x2 t-SNE comparison plot showing Raw Input vs Sample Encoding,
    colored by temperature (top row), pressure (middle row), and XFA_in (bottom row).

    Args:
        representations_dict: dict mapping representation name to features array
        output_dir: Directory to save plots
        temp_labels: Temperature values for coloring
        pressure_labels: Pressure values for coloring
        xfa_labels: Initial concentration (XFA_in) values for coloring
    """
    perplexity = 30
    n_iter = 1500
    init_method = 'pca'

    # Determine number of rows based on whether xfa_labels is provided
    n_rows = 3 if xfa_labels is not None else 2

    try:
        fig, axes = plt.subplots(n_rows, 2, figsize=(14, 6 * n_rows))

        # Get representations
        raw_features = representations_dict['Raw Input (Standardized)']
        sample_features = representations_dict['Sample Encoding']

        # Standardize features
        scaler_raw = StandardScaler()
        scaler_sample = StandardScaler()
        raw_scaled = scaler_raw.fit_transform(raw_features)
        sample_scaled = scaler_sample.fit_transform(sample_features)

        # Compute t-SNE for both representations
        logger.info(f"Computing t-SNE with perplexity={perplexity}...")

        tsne_raw = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                        n_iter=n_iter, init=init_method, learning_rate='auto')
        raw_2d = tsne_raw.fit_transform(raw_scaled)

        tsne_sample = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                           n_iter=n_iter, init=init_method, learning_rate='auto')
        sample_2d = tsne_sample.fit_transform(sample_scaled)

        # Row 1: Temperature
        # Top-left: Raw input colored by temperature
        ax = axes[0, 0]
        scatter = ax.scatter(raw_2d[:, 0], raw_2d[:, 1],
                            c=temp_labels, cmap='plasma', alpha=0.7, s=60, edgecolors='white', linewidths=0.5)
        plt.colorbar(scatter, ax=ax, label='Temperature (K)')
        ax.set_xlabel('t-SNE 1', fontsize=12)
        ax.set_ylabel('t-SNE 2', fontsize=12)
        ax.set_title('Raw Input (Standardized)\nColored by Temperature', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Top-right: Sample encoding colored by temperature
        ax = axes[0, 1]
        scatter = ax.scatter(sample_2d[:, 0], sample_2d[:, 1],
                            c=temp_labels, cmap='plasma', alpha=0.7, s=60, edgecolors='white', linewidths=0.5)
        plt.colorbar(scatter, ax=ax, label='Temperature (K)')
        ax.set_xlabel('t-SNE 1', fontsize=12)
        ax.set_ylabel('t-SNE 2', fontsize=12)
        ax.set_title('Sample Encoding\nColored by Temperature', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Row 2: Pressure
        # Middle-left: Raw input colored by pressure
        ax = axes[1, 0]
        scatter = ax.scatter(raw_2d[:, 0], raw_2d[:, 1],
                            c=pressure_labels, cmap='viridis', alpha=0.7, s=60, edgecolors='white', linewidths=0.5)
        plt.colorbar(scatter, ax=ax, label='Pressure (bar)')
        ax.set_xlabel('t-SNE 1', fontsize=12)
        ax.set_ylabel('t-SNE 2', fontsize=12)
        ax.set_title('Raw Input (Standardized)\nColored by Pressure', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Middle-right: Sample encoding colored by pressure
        ax = axes[1, 1]
        scatter = ax.scatter(sample_2d[:, 0], sample_2d[:, 1],
                            c=pressure_labels, cmap='viridis', alpha=0.7, s=60, edgecolors='white', linewidths=0.5)
        plt.colorbar(scatter, ax=ax, label='Pressure (bar)')
        ax.set_xlabel('t-SNE 1', fontsize=12)
        ax.set_ylabel('t-SNE 2', fontsize=12)
        ax.set_title('Sample Encoding\nColored by Pressure', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Row 3: XFA_in (if provided)
        if xfa_labels is not None:
            # Bottom-left: Raw input colored by XFA_in
            ax = axes[2, 0]
            scatter = ax.scatter(raw_2d[:, 0], raw_2d[:, 1],
                                c=xfa_labels, cmap='coolwarm', alpha=0.7, s=60, edgecolors='white', linewidths=0.5)
            plt.colorbar(scatter, ax=ax, label='XFA_in (Initial Conc.)')
            ax.set_xlabel('t-SNE 1', fontsize=12)
            ax.set_ylabel('t-SNE 2', fontsize=12)
            ax.set_title('Raw Input (Standardized)\nColored by Initial Concentration', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Bottom-right: Sample encoding colored by XFA_in
            ax = axes[2, 1]
            scatter = ax.scatter(sample_2d[:, 0], sample_2d[:, 1],
                                c=xfa_labels, cmap='coolwarm', alpha=0.7, s=60, edgecolors='white', linewidths=0.5)
            plt.colorbar(scatter, ax=ax, label='XFA_in (Initial Conc.)')
            ax.set_xlabel('t-SNE 1', fontsize=12)
            ax.set_ylabel('t-SNE 2', fontsize=12)
            ax.set_title('Sample Encoding\nColored by Initial Concentration', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)

        plt.suptitle(f't-SNE Visualization: Raw Input vs Sample Encoding\n'
                     f'(perplexity={perplexity}, {n_iter} iterations, PCA init)',
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        save_path = os.path.join(output_dir, 'tsne_raw_vs_encoding.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved t-SNE raw vs encoding plot to {save_path}")

    except Exception as e:
        logger.error(f"Failed to create t-SNE raw vs encoding plot: {e}")
        plt.close('all')


def plot_pca_raw_vs_encoding(representations_dict, output_dir, temp_labels, pressure_labels, xfa_labels=None):
    """
    Create a 2x2 PCA comparison plot showing Raw Input vs Sample Encoding,
    colored by temperature (top row) and XFA_in (bottom row).

    PCA preserves global structure better than t-SNE, showing linear relationships.

    Args:
        representations_dict: dict mapping representation name to features array
        output_dir: Directory to save plots
        temp_labels: Temperature values for coloring
        pressure_labels: Pressure values for coloring (not used in plot, kept for API compatibility)
        xfa_labels: Initial concentration (XFA_in) values for coloring
    """
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Get representations
        raw_features = representations_dict['Raw Input (Standardized)']
        sample_features = representations_dict['Sample Encoding']

        # Standardize features before PCA
        scaler_raw = StandardScaler()
        scaler_sample = StandardScaler()
        raw_scaled = scaler_raw.fit_transform(raw_features)
        sample_scaled = scaler_sample.fit_transform(sample_features)

        # Compute PCA for both representations
        logger.info("Computing PCA...")

        pca_raw = PCA(n_components=2)
        raw_2d = pca_raw.fit_transform(raw_scaled)
        raw_var_explained = pca_raw.explained_variance_ratio_

        pca_sample = PCA(n_components=2)
        sample_2d = pca_sample.fit_transform(sample_scaled)
        sample_var_explained = pca_sample.explained_variance_ratio_

        logger.info(f"Raw Input - Variance explained: PC1={raw_var_explained[0]:.3f}, PC2={raw_var_explained[1]:.3f}, Total={sum(raw_var_explained):.3f}")
        logger.info(f"Sample Encoding - Variance explained: PC1={sample_var_explained[0]:.3f}, PC2={sample_var_explained[1]:.3f}, Total={sum(sample_var_explained):.3f}")

        # Row 1: Temperature
        # Top-left: Raw input colored by temperature
        ax = axes[0, 0]
        scatter = ax.scatter(raw_2d[:, 0], raw_2d[:, 1],
                            c=temp_labels, cmap='plasma', alpha=0.7, s=60, edgecolors='white', linewidths=0.5)
        plt.colorbar(scatter, ax=ax, label='Temperature (K)')
        ax.set_xlabel(f'PC1 ({raw_var_explained[0]*100:.1f}%)', fontsize=12)
        ax.set_ylabel(f'PC2 ({raw_var_explained[1]*100:.1f}%)', fontsize=12)
        ax.set_title('Raw Input (Standardized)\nColored by Temperature', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Top-right: Sample encoding colored by temperature
        ax = axes[0, 1]
        scatter = ax.scatter(sample_2d[:, 0], sample_2d[:, 1],
                            c=temp_labels, cmap='plasma', alpha=0.7, s=60, edgecolors='white', linewidths=0.5)
        plt.colorbar(scatter, ax=ax, label='Temperature (K)')
        ax.set_xlabel(f'PC1 ({sample_var_explained[0]*100:.1f}%)', fontsize=12)
        ax.set_ylabel(f'PC2 ({sample_var_explained[1]*100:.1f}%)', fontsize=12)
        ax.set_title('Sample Encoding\nColored by Temperature', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Row 2: XFA_in (Initial Concentration)
        # Bottom-left: Raw input colored by XFA_in
        ax = axes[1, 0]
        if xfa_labels is not None:
            scatter = ax.scatter(raw_2d[:, 0], raw_2d[:, 1],
                                c=xfa_labels, cmap='coolwarm', alpha=0.7, s=60, edgecolors='white', linewidths=0.5)
            plt.colorbar(scatter, ax=ax, label='XFA_in (Initial Conc.)')
        else:
            ax.scatter(raw_2d[:, 0], raw_2d[:, 1], alpha=0.7, s=60, c='#2E86AB')
        ax.set_xlabel(f'PC1 ({raw_var_explained[0]*100:.1f}%)', fontsize=12)
        ax.set_ylabel(f'PC2 ({raw_var_explained[1]*100:.1f}%)', fontsize=12)
        ax.set_title('Raw Input (Standardized)\nColored by Initial Concentration', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Bottom-right: Sample encoding colored by XFA_in
        ax = axes[1, 1]
        if xfa_labels is not None:
            scatter = ax.scatter(sample_2d[:, 0], sample_2d[:, 1],
                                c=xfa_labels, cmap='coolwarm', alpha=0.7, s=60, edgecolors='white', linewidths=0.5)
            plt.colorbar(scatter, ax=ax, label='XFA_in (Initial Conc.)')
        else:
            ax.scatter(sample_2d[:, 0], sample_2d[:, 1], alpha=0.7, s=60, c='#2E86AB')
        ax.set_xlabel(f'PC1 ({sample_var_explained[0]*100:.1f}%)', fontsize=12)
        ax.set_ylabel(f'PC2 ({sample_var_explained[1]*100:.1f}%)', fontsize=12)
        ax.set_title('Sample Encoding\nColored by Initial Concentration', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add variance explained summary
        raw_total = sum(raw_var_explained) * 100
        sample_total = sum(sample_var_explained) * 100
        plt.suptitle(f'PCA Visualization: Raw Input vs Sample Encoding\n'
                     f'(Raw: {raw_total:.1f}% variance explained, Sample Encoding: {sample_total:.1f}% variance explained)',
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        save_path = os.path.join(output_dir, 'pca_raw_vs_encoding.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved PCA raw vs encoding plot to {save_path}")

        # Return variance explained for reporting
        return {
            'raw': {'pc1': raw_var_explained[0], 'pc2': raw_var_explained[1], 'total': sum(raw_var_explained)},
            'sample': {'pc1': sample_var_explained[0], 'pc2': sample_var_explained[1], 'total': sum(sample_var_explained)}
        }

    except Exception as e:
        logger.error(f"Failed to create PCA raw vs encoding plot: {e}")
        plt.close('all')
        return None


def plot_separability_improvement(distances_dict, output_dir):
    """
    Plot showing how separability improves through the encoding pipeline.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate statistics
    names = list(distances_dict.keys())
    means = [np.mean(d) for d in distances_dict.values()]
    stds = [np.std(d) for d in distances_dict.values()]
    cvs = [s/m if m != 0 else 0 for m, s in zip(means, stds)]

    x = np.arange(len(names))
    width = 0.35

    # Plot mean distance and CV
    ax2 = ax.twinx()

    bars1 = ax.bar(x - width/2, means, width, label='Mean Distance', color='#2E86AB', alpha=0.7)
    bars2 = ax2.bar(x + width/2, cvs, width, label='CV (Std/Mean)', color='#F18F01', alpha=0.7)

    ax.set_xlabel('Representation Stage', fontsize=12)
    ax.set_ylabel('Mean Pairwise Distance', fontsize=12, color='#2E86AB')
    ax2.set_ylabel('Coefficient of Variation', fontsize=12, color='#F18F01')

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.tick_params(axis='y', labelcolor='#2E86AB')
    ax2.tick_params(axis='y', labelcolor='#F18F01')

    # Add value labels
    for bar, val in zip(bars1, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.3f}',
                ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, cvs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.3f}',
                 ha='center', va='bottom', fontsize=9)

    ax.set_title('Separability Metrics Through Encoding Pipeline',
                 fontsize=14, fontweight='bold')

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    save_path = os.path.join(output_dir, 'separability_improvement.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved separability improvement plot to {save_path}")


def save_numerical_results(representations_dict, distances_euclidean, distances_cosine, output_dir):
    """
    Save numerical results to JSON and CSV files.
    """
    results = {
        'euclidean_distances': {},
        'cosine_distances': {},
        'normalized_distances': {},
        'summary_statistics': {}
    }

    for name in representations_dict.keys():
        n_features = representations_dict[name].shape[1]

        # Euclidean
        euc_dists = distances_euclidean[name]
        mean_euc = float(np.mean(euc_dists)) if len(euc_dists) > 0 else 0
        std_euc = float(np.std(euc_dists)) if len(euc_dists) > 0 else 0
        results['euclidean_distances'][name] = {
            'mean': mean_euc,
            'std': std_euc,
            'min': float(np.min(euc_dists)) if len(euc_dists) > 0 else 0,
            'max': float(np.max(euc_dists)) if len(euc_dists) > 0 else 0,
            'median': float(np.median(euc_dists)) if len(euc_dists) > 0 else 0,
            'cv': float(std_euc / mean_euc) if mean_euc != 0 else 0
        }

        # Normalized Euclidean (per-dimension average)
        # This allows fair comparison across different feature dimensions
        normalized_mean = mean_euc / np.sqrt(n_features) if n_features > 0 else 0
        normalized_std = std_euc / np.sqrt(n_features) if n_features > 0 else 0
        results['normalized_distances'][name] = {
            'normalized_mean': float(normalized_mean),
            'normalized_std': float(normalized_std),
            'n_features': int(n_features),
            'interpretation': 'Euclidean distance divided by sqrt(n_features) for fair comparison'
        }

        # Cosine
        cos_dists = distances_cosine[name]
        mean_cos = float(np.mean(cos_dists)) if len(cos_dists) > 0 else 0
        std_cos = float(np.std(cos_dists)) if len(cos_dists) > 0 else 0
        results['cosine_distances'][name] = {
            'mean': mean_cos,
            'std': std_cos,
            'min': float(np.min(cos_dists)) if len(cos_dists) > 0 else 0,
            'max': float(np.max(cos_dists)) if len(cos_dists) > 0 else 0,
            'median': float(np.median(cos_dists)) if len(cos_dists) > 0 else 0,
            'cv': float(std_cos / mean_cos) if mean_cos != 0 else 0
        }

        # Feature dimensions
        results['summary_statistics'][name] = {
            'n_samples': int(representations_dict[name].shape[0]),
            'n_features': int(n_features)
        }

    # Calculate improvement metrics
    raw_stats = results['euclidean_distances']['Raw Input (Standardized)']
    sample_stats = results['euclidean_distances']['Sample Encoding']
    fused_stats = results['euclidean_distances']['Fused Features']

    raw_norm = results['normalized_distances']['Raw Input (Standardized)']['normalized_mean']
    sample_norm = results['normalized_distances']['Sample Encoding']['normalized_mean']
    fused_norm = results['normalized_distances']['Fused Features']['normalized_mean']

    results['improvement_metrics'] = {
        'euclidean': {
            'raw_to_sample_ratio': float(sample_stats['mean'] / raw_stats['mean']) if raw_stats['mean'] != 0 else 0,
            'raw_to_fused_ratio': float(fused_stats['mean'] / raw_stats['mean']) if raw_stats['mean'] != 0 else 0,
        },
        'normalized_euclidean': {
            'raw_to_sample_ratio': float(sample_norm / raw_norm) if raw_norm != 0 else 0,
            'raw_to_fused_ratio': float(fused_norm / raw_norm) if raw_norm != 0 else 0,
            'interpretation': 'Ratios > 1 indicate increased separability; normalized for fair feature dimension comparison'
        },
        'coefficient_of_variation': {
            'raw_cv': float(raw_stats['cv']),
            'sample_cv': float(sample_stats['cv']),
            'fused_cv': float(fused_stats['cv']),
            'interpretation': 'Higher CV indicates more diverse distance distribution (potentially better discrimination)'
        },
        'key_findings': {
            'sample_encoding_increases_spread': float(sample_stats['mean'] > raw_stats['mean']),
            'fused_features_higher_cv': float(fused_stats['cv'] > raw_stats['cv']),
            'summary': 'Sample Encoding spreads samples apart; Fusion adds variability while combining time+sample info'
        }
    }

    # Save JSON
    json_path = os.path.join(output_dir, 'encoding_analysis_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved numerical results to {json_path}")

    # Save summary CSV
    summary_data = []
    for name in representations_dict.keys():
        row = {
            'Representation': name,
            'N_Features': results['summary_statistics'][name]['n_features'],
            'Euclidean_Mean': results['euclidean_distances'][name]['mean'],
            'Euclidean_Std': results['euclidean_distances'][name]['std'],
            'Euclidean_CV': results['euclidean_distances'][name]['cv'],
            'Normalized_Mean': results['normalized_distances'][name]['normalized_mean'],
            'Normalized_Std': results['normalized_distances'][name]['normalized_std'],
            'Cosine_Mean': results['cosine_distances'][name]['mean'],
            'Cosine_Std': results['cosine_distances'][name]['std'],
            'Cosine_CV': results['cosine_distances'][name]['cv']
        }
        summary_data.append(row)

    df_summary = pd.DataFrame(summary_data)
    csv_path = os.path.join(output_dir, 'encoding_analysis_summary.csv')
    df_summary.to_csv(csv_path, index=False)
    logger.info(f"Saved summary CSV to {csv_path}")

    return results
