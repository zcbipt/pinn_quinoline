# ------------------------------------------------------------------------------------------
# Description: Analyze how measurement noise in training data affects learned chemical parameters
#              - Demonstrates indirect influence pathway: data noise → NN predictions → PDE loss → parameters
#              - Uses noise injection and gradient propagation analysis
# ------------------------------------------------------------------------------------------

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from loguru import logger
from collections import defaultdict
import copy


from src.model_pinn import PINN
from src.loss import compute_pde_loss


def get_chemical_parameters(model):
    """Extract chemical model parameters (x_A_*, y_E_*, z_Delta_H_*)."""
    chemical_params = {}
    for name, param in model.named_parameters():
        if any(prefix in name for prefix in ['x_A_', 'y_E_', 'z_Delta_H_']):
            chemical_params[name] = param
    return chemical_params


def get_parameter_values(model):
    """Get current values of all chemical parameters."""
    params = get_chemical_parameters(model)
    return {name: param.detach().clone() for name, param in params.items()}


def simulate_training_step(model, optimizer, input_features, targets_scaled,
                           available_indices, env_dict, loss_weights=None):
    """
    Simulate a single training step and return the parameter updates.

    Args:
        model: PINN model
        optimizer: Optimizer instance
        input_features: Input tensor
        targets_scaled: Target tensor
        available_indices: Indices of available outputs
        env_dict: Environment dictionary
        loss_weights: Optional dict of loss weights

    Returns:
        dict: Parameter updates (new_value - old_value) for each chemical parameter
    """
    if loss_weights is None:
        loss_weights = {'data': 1.0, 'pde': 1.0}

    # Record old parameter values
    chemical_params = get_chemical_parameters(model)
    old_values = {name: param.detach().clone() for name, param in chemical_params.items()}

    # Forward pass
    c_pred, time_points = model(input_features)
    x_pred, env_pred = model.process_c_pred(c_pred=c_pred, env_dict=env_dict,
                                            time_points=time_points)

    # Compute losses
    predictions = c_pred[:, -1, :]
    predictions_available = predictions[:, available_indices]
    data_loss = torch.mean((predictions_available - targets_scaled) ** 2)

    pde_residuals = compute_pde_loss(model, c_initial=input_features,
                                     time_points=time_points, env_pred=env_pred)
    pde_loss = sum(torch.mean(r ** 2) for r in pde_residuals) / len(pde_residuals)

    total_loss = loss_weights['data'] * data_loss + loss_weights['pde'] * pde_loss

    # Backward and update
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # Compute parameter updates
    updates = {}
    for name, param in chemical_params.items():
        update = param.detach() - old_values[name]
        updates[name] = {
            'update': update.item() if update.numel() == 1 else update.tolist(),
            'update_norm': torch.norm(update).item(),
            'relative_update': (torch.norm(update) / (torch.norm(old_values[name]) + 1e-10)).item()
        }

    return updates, data_loss.item(), pde_loss.item()


def analyze_gradient_propagation(model, df_lab, config, device, noise_levels):
    """
    Analyze how data perturbations propagate through to chemical parameter gradients
    using mini-training simulation.

    This demonstrates the indirect pathway:
    Noisy data → Different NN updates → Different predictions → Different PDE residuals
    → Different chemical parameter updates

    Args:
        model: Trained PINN model
        df_lab: Original laboratory data
        config: Configuration
        device: Torch device
        noise_levels: List of noise levels to test (e.g., [0.01, 0.05, 0.10])

    Returns:
        dict: Results showing gradient changes at each noise level
    """
    c_scaling_factor = config['config_model']['c_scaling_factor']

    # Prepare base data
    input_features, targets_scaled, initial_conc, available_indices, env_dict = \
        prepare_data_tensors(df_lab, config, device)

    # Store original model state
    original_state = copy.deepcopy(model.state_dict())

    # Get baseline gradient norms (no training, just compute gradients)
    model.train()
    for param in model.parameters():
        param.requires_grad_(True)

    input_features.requires_grad_(True)

    # Forward pass with original data
    c_pred_base, time_points = model(input_features)
    x_pred, env_pred = model.process_c_pred(c_pred=c_pred_base, env_dict=env_dict,
                                             time_points=time_points)

    # Compute PDE loss
    pde_residuals = compute_pde_loss(model, c_initial=input_features,
                                     time_points=time_points, env_pred=env_pred)
    pde_loss_base = sum(torch.mean(r ** 2) for r in pde_residuals) / len(pde_residuals)

    # Also compute data loss
    predictions = c_pred_base[:, -1, :]
    predictions_available = predictions[:, available_indices]
    data_loss_base = torch.mean((predictions_available - targets_scaled) ** 2)

    # Total loss (as in training)
    total_loss_base = data_loss_base + pde_loss_base

    # Compute baseline gradients
    model.zero_grad()
    total_loss_base.backward(retain_graph=True)

    chemical_params = get_chemical_parameters(model)
    baseline_gradients = {}
    for name, param in chemical_params.items():
        if param.grad is not None:
            baseline_gradients[name] = param.grad.clone().detach()
        else:
            baseline_gradients[name] = torch.zeros_like(param)

    logger.info(f"Baseline PDE loss: {pde_loss_base.item():.6f}")
    logger.info(f"Baseline Data loss: {data_loss_base.item():.6f}")

    # Log baseline gradient norms
    total_grad_norm = sum(torch.norm(g).item() for g in baseline_gradients.values())
    logger.info(f"Total baseline gradient norm: {total_grad_norm:.6f}")

    results = {
        'noise_levels': noise_levels,
        'baseline': {
            'pde_loss': pde_loss_base.item(),
            'data_loss': data_loss_base.item(),
            'total_loss': total_loss_base.item(),
            'gradient_norms': {name: torch.norm(g).item() for name, g in baseline_gradients.items()}
        },
        'noisy_results': []
    }

    # Restore model state
    model.load_state_dict(original_state)

    # Test each noise level using mini-training simulation
    n_steps = 5  # Number of training steps to simulate
    lr = 1e-4  # Small learning rate for simulation

    for noise_level in noise_levels:
        logger.info(f"\nAnalyzing noise level: {noise_level*100:.1f}%")

        # --- Baseline training (clean data) ---
        model.load_state_dict(original_state)
        optimizer_clean = torch.optim.Adam(model.parameters(), lr=lr)

        clean_updates_total = defaultdict(float)
        for step in range(n_steps):
            updates_clean, _, _ = simulate_training_step(
                model, optimizer_clean, input_features, targets_scaled,
                available_indices, env_dict
            )
            for name, data in updates_clean.items():
                clean_updates_total[name] += abs(data['update_norm'])

        # --- Noisy training ---
        model.load_state_dict(original_state)
        optimizer_noisy = torch.optim.Adam(model.parameters(), lr=lr)

        # Add noise to targets
        noise = torch.randn_like(targets_scaled) * targets_scaled.abs() * noise_level
        targets_noisy = targets_scaled + noise

        noisy_updates_total = defaultdict(float)
        for step in range(n_steps):
            updates_noisy, _, _ = simulate_training_step(
                model, optimizer_noisy, input_features, targets_noisy,
                available_indices, env_dict
            )
            for name, data in updates_noisy.items():
                noisy_updates_total[name] += abs(data['update_norm'])

        # Compute differences
        gradient_changes = {}
        for name in chemical_params:
            clean_update = clean_updates_total[name]
            noisy_update = noisy_updates_total[name]
            diff = abs(noisy_update - clean_update)

            if clean_update > 1e-10:
                relative_change = diff / clean_update * 100
            else:
                relative_change = 0 if diff < 1e-10 else float('inf')

            gradient_changes[name] = {
                'clean_update_norm': clean_update,
                'noisy_update_norm': noisy_update,
                'diff_norm': diff,
                'relative_change_pct': relative_change
            }

        # Compute final losses after noisy training
        model.eval()
        with torch.no_grad():
            c_pred_final, time_points_final = model(input_features)
            x_pred_f, env_pred_f = model.process_c_pred(
                c_pred=c_pred_final, env_dict=env_dict, time_points=time_points_final
            )

        model.train()
        c_pred_final, time_points_final = model(input_features)
        x_pred_f, env_pred_f = model.process_c_pred(
            c_pred=c_pred_final, env_dict=env_dict, time_points=time_points_final
        )

        pde_residuals_f = compute_pde_loss(model, c_initial=input_features,
                                           time_points=time_points_final, env_pred=env_pred_f)
        pde_loss_final = sum(torch.mean(r ** 2) for r in pde_residuals_f) / len(pde_residuals_f)

        predictions_f = c_pred_final[:, -1, :]
        predictions_available_f = predictions_f[:, available_indices]
        data_loss_final = torch.mean((predictions_available_f - targets_scaled) ** 2)

        results['noisy_results'].append({
            'noise_level': noise_level,
            'pde_loss': pde_loss_final.item(),
            'data_loss': data_loss_final.item(),
            'total_loss': (data_loss_final + pde_loss_final).item(),
            'gradient_changes': gradient_changes
        })

        # Log summary
        valid_changes = [gc['relative_change_pct'] for gc in gradient_changes.values()
                        if gc['relative_change_pct'] < float('inf')]
        avg_change = np.mean(valid_changes) if valid_changes else 0

        logger.info(f"  PDE loss: {pde_loss_final.item():.6f} (Δ = {pde_loss_final.item() - pde_loss_base.item():+.6f})")
        logger.info(f"  Data loss: {data_loss_final.item():.6f} (Δ = {data_loss_final.item() - data_loss_base.item():+.6f})")
        logger.info(f"  Avg parameter update change: {avg_change:.2f}%")

    # Restore original model state
    model.load_state_dict(original_state)

    return results


def analyze_per_sample_influence(model, df_lab, config, device, n_samples=10):
    """
    Analyze which data samples most influence the chemical parameters.

    Uses leave-one-out analysis on a subset of samples.

    Args:
        model: Trained PINN model
        df_lab: Laboratory data
        config: Configuration
        device: Torch device
        n_samples: Number of samples to analyze (randomly selected)

    Returns:
        dict: Influence scores for each sample
    """
    c_scaling_factor = config['config_model']['c_scaling_factor']

    # Select random samples to analyze
    n_total = len(df_lab)
    sample_indices = np.random.choice(n_total, min(n_samples, n_total), replace=False)

    results = {
        'sample_indices': sample_indices.tolist(),
        'influence_scores': {}
    }

    # Get baseline gradients with all data
    input_features_full, targets_scaled_full, initial_conc_full, available_indices, env_dict_full = \
        prepare_data_tensors(df_lab, config, device)

    model.train()
    for param in model.parameters():
        param.requires_grad_(True)

    input_features_full.requires_grad_(True)

    # Forward pass
    c_pred_full, time_points = model(input_features_full)
    x_pred, env_pred = model.process_c_pred(c_pred=c_pred_full, env_dict=env_dict_full,
                                            time_points=time_points)

    # Compute losses
    pde_residuals = compute_pde_loss(model, c_initial=input_features_full,
                                     time_points=time_points, env_pred=env_pred)
    pde_loss = sum(torch.mean(r ** 2) for r in pde_residuals) / len(pde_residuals)

    predictions = c_pred_full[:, -1, :]
    predictions_available = predictions[:, available_indices]
    data_loss = torch.mean((predictions_available - targets_scaled_full) ** 2)

    total_loss = data_loss + pde_loss

    # Compute baseline gradients
    model.zero_grad()
    total_loss.backward(retain_graph=True)

    chemical_params = get_chemical_parameters(model)
    baseline_gradients = {}
    for name, param in chemical_params.items():
        if param.grad is not None:
            baseline_gradients[name] = param.grad.clone().detach()
        else:
            baseline_gradients[name] = torch.zeros_like(param)

    baseline_total_norm = sum(torch.norm(g).item() for g in baseline_gradients.values())

    # Analyze each sample
    for idx in sample_indices:
        logger.info(f"Analyzing sample {idx}...")

        # Create mask to exclude this sample
        mask = torch.ones(n_total, dtype=torch.bool, device=device)
        mask[idx] = False

        # Apply mask to tensors
        input_features_loo = input_features_full[mask]
        targets_scaled_loo = targets_scaled_full[mask]

        # Update env_dict for leave-one-out
        env_dict_loo = {k: v[mask] for k, v in env_dict_full.items()}

        # Forward pass
        input_features_loo.requires_grad_(True)
        c_pred_loo, time_points_loo = model(input_features_loo)
        x_pred_loo, env_pred_loo = model.process_c_pred(
            c_pred=c_pred_loo, env_dict=env_dict_loo, time_points=time_points_loo
        )

        # Compute losses
        pde_residuals_loo = compute_pde_loss(model, c_initial=input_features_loo,
                                             time_points=time_points_loo, env_pred=env_pred_loo)
        pde_loss_loo = sum(torch.mean(r ** 2) for r in pde_residuals_loo) / len(pde_residuals_loo)

        predictions_loo = c_pred_loo[:, -1, :]
        predictions_available_loo = predictions_loo[:, available_indices]
        data_loss_loo = torch.mean((predictions_available_loo - targets_scaled_loo) ** 2)

        total_loss_loo = data_loss_loo + pde_loss_loo

        # Compute gradients
        model.zero_grad()
        total_loss_loo.backward(retain_graph=True)

        loo_gradients = {}
        for name, param in chemical_params.items():
            if param.grad is not None:
                loo_gradients[name] = param.grad.clone().detach()
            else:
                loo_gradients[name] = torch.zeros_like(param)

        # Compute influence as change in gradient when sample is removed
        influence = {}
        total_influence = 0
        for name in chemical_params:
            diff = torch.norm(baseline_gradients[name] - loo_gradients[name]).item()
            influence[name] = diff
            total_influence += diff

        results['influence_scores'][int(idx)] = {
            'total_influence': total_influence,
            'per_parameter': influence,
            'sample_info': {
                'T': df_lab.iloc[idx]['T'],
                'P': df_lab.iloc[idx]['P'],
                'XFA_in': df_lab.iloc[idx]['XFA_in']
            }
        }

    # Rank samples by influence
    ranked = sorted(results['influence_scores'].items(),
                   key=lambda x: x[1]['total_influence'], reverse=True)
    results['ranking'] = [{'sample_idx': idx, 'influence': data['total_influence']}
                         for idx, data in ranked]

    return results


def compute_noise_amplification_factor(model, df_lab, config, device, noise_level=0.05, n_trials=5):
    """
    Compute how measurement noise amplifies through to parameter updates.

    Uses mini-training simulation to properly capture the noise propagation pathway.
    Amplification Factor = (% change in parameter update) / (% noise in data)

    Args:
        model: Trained PINN model
        df_lab: Laboratory data
        config: Configuration
        device: Torch device
        noise_level: Noise level to test
        n_trials: Number of random trials to average

    Returns:
        dict: Amplification factors for each parameter
    """
    c_scaling_factor = config['config_model']['c_scaling_factor']

    input_features, targets_scaled, initial_conc, available_indices, env_dict = \
        prepare_data_tensors(df_lab, config, device)

    chemical_params = get_chemical_parameters(model)

    # Store original model state
    original_state = copy.deepcopy(model.state_dict())

    # Parameters for training simulation
    n_steps = 5
    lr = 1e-4

    update_changes = defaultdict(list)

    for trial in range(n_trials):
        logger.info(f"  Trial {trial+1}/{n_trials}")

        # --- Baseline training (clean data) ---
        model.load_state_dict(original_state)
        optimizer_clean = torch.optim.Adam(model.parameters(), lr=lr)

        clean_updates_total = defaultdict(float)
        for step in range(n_steps):
            updates_clean, _, _ = simulate_training_step(
                model, optimizer_clean, input_features, targets_scaled,
                available_indices, env_dict
            )
            for name, data in updates_clean.items():
                clean_updates_total[name] += abs(data['update_norm'])

        # --- Noisy training ---
        model.load_state_dict(original_state)
        optimizer_noisy = torch.optim.Adam(model.parameters(), lr=lr)

        # Add random noise to targets
        noise = torch.randn_like(targets_scaled) * targets_scaled.abs() * noise_level
        targets_noisy = targets_scaled + noise

        noisy_updates_total = defaultdict(float)
        for step in range(n_steps):
            updates_noisy, _, _ = simulate_training_step(
                model, optimizer_noisy, input_features, targets_noisy,
                available_indices, env_dict
            )
            for name, data in updates_noisy.items():
                noisy_updates_total[name] += abs(data['update_norm'])

        # Compute relative change for each parameter
        for name in chemical_params:
            clean_update = clean_updates_total[name]
            noisy_update = noisy_updates_total[name]

            if clean_update > 1e-10:
                relative_change = abs(noisy_update - clean_update) / clean_update
                update_changes[name].append(relative_change)

    # Restore original model state
    model.load_state_dict(original_state)

    # Compute amplification factors
    results = {
        'noise_level': noise_level,
        'n_trials': n_trials,
        'amplification_factors': {}
    }

    for name in chemical_params:
        if update_changes[name]:
            mean_change = np.mean(update_changes[name])
            std_change = np.std(update_changes[name])

            # Amplification = (update change %) / (data noise %)
            amplification = mean_change / noise_level

            results['amplification_factors'][name] = {
                'mean_relative_change': mean_change,
                'std_relative_change': std_change,
                'amplification_factor': amplification
            }
        else:
            # If no valid measurements, set to 0
            results['amplification_factors'][name] = {
                'mean_relative_change': 0,
                'std_relative_change': 0,
                'amplification_factor': 0
            }

    return results


def plot_gradient_propagation(results, output_dir):
    """Plot gradient propagation analysis results."""
    noise_levels = [r['noise_level'] * 100 for r in results['noisy_results']]

    # Collect data for plotting - handle both old and new key names
    param_changes = defaultdict(list)
    for noisy_result in results['noisy_results']:
        for name, data in noisy_result['gradient_changes'].items():
            change_pct = data.get('relative_change_pct', 0)
            if change_pct < float('inf'):
                param_changes[name].append(change_pct)
            else:
                param_changes[name].append(0)

    # Group by parameter type
    param_groups = {
        'x_A (Pre-exponential)': [],
        'y_E (Activation energy)': [],
        'z_Delta_H (Enthalpy)': []
    }

    for name in param_changes:
        if 'x_A_' in name:
            param_groups['x_A (Pre-exponential)'].append(name)
        elif 'y_E_' in name:
            param_groups['y_E (Activation energy)'].append(name)
        elif 'z_Delta_H_' in name:
            param_groups['z_Delta_H (Enthalpy)'].append(name)

    # Plot 1: Average gradient change by parameter type
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'x_A (Pre-exponential)': '#3498DB',
              'y_E (Activation energy)': '#E74C3C',
              'z_Delta_H (Enthalpy)': '#2ECC71'}

    for group_name, param_names in param_groups.items():
        if param_names:
            avg_changes = []
            for i in range(len(noise_levels)):
                changes = [param_changes[name][i] for name in param_names]
                avg_changes.append(np.mean(changes))

            ax.plot(noise_levels, avg_changes, 'o-', label=group_name,
                   color=colors[group_name], linewidth=2, markersize=8)

    ax.set_xlabel('Measurement Noise Level (%)', fontsize=12)
    ax.set_ylabel('Gradient Change (%)', fontsize=12)
    ax.set_title('Effect of Measurement Noise on Chemical Parameter Gradients\n'
                '(Demonstrates indirect pathway: noise → predictions → PDE loss → gradients)',
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add reference line for 1:1 relationship
    ax.plot(noise_levels, noise_levels, 'k--', alpha=0.5, label='1:1 reference')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'gradient_propagation.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved gradient propagation plot to {save_path}")

    # Plot 2: Loss changes with noise
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    pde_losses = [results['baseline']['pde_loss']] + \
                 [r['pde_loss'] for r in results['noisy_results']]
    data_losses = [results['baseline']['data_loss']] + \
                  [r['data_loss'] for r in results['noisy_results']]
    all_noise_levels = [0] + noise_levels

    ax = axes[0]
    ax.plot(all_noise_levels, pde_losses, 'o-', color='#E74C3C',
            linewidth=2, markersize=8, label='PDE Loss')
    ax.set_xlabel('Measurement Noise Level (%)', fontsize=11)
    ax.set_ylabel('PDE Loss', fontsize=11)
    ax.set_title('PDE Loss vs Noise Level', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(all_noise_levels, data_losses, 'o-', color='#3498DB',
            linewidth=2, markersize=8, label='Data Loss')
    ax.set_xlabel('Measurement Noise Level (%)', fontsize=11)
    ax.set_ylabel('Data Loss', fontsize=11)
    ax.set_title('Data Loss vs Noise Level', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.suptitle('Impact of Measurement Noise on Loss Components',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = os.path.join(output_dir, 'loss_vs_noise.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved loss vs noise plot to {save_path}")


def plot_amplification_factors(results, output_dir):
    """Plot noise amplification factors."""
    factors = results['amplification_factors']

    # Sort by amplification factor
    sorted_params = sorted(factors.items(), key=lambda x: x[1]['amplification_factor'], reverse=True)

    names = [p[0].split('.')[-1] if '.' in p[0] else p[0] for p in sorted_params]
    amplifications = [p[1]['amplification_factor'] for p in sorted_params]

    # Color by type
    colors = []
    for p in sorted_params:
        name = p[0]
        if 'x_A_' in name:
            colors.append('#3498DB')
        elif 'y_E_' in name:
            colors.append('#E74C3C')
        else:
            colors.append('#2ECC71')

    fig, ax = plt.subplots(figsize=(14, 8))

    bars = ax.barh(range(len(names)), amplifications, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Noise Amplification Factor', fontsize=12)
    ax.set_title(f'Measurement Noise Amplification by Parameter\n'
                f'(Factor = % gradient change / % data noise, noise level = {results["noise_level"]*100:.0f}%)',
                fontsize=12, fontweight='bold')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498DB', label='x_A (Pre-exponential)'),
        Patch(facecolor='#E74C3C', label='y_E (Activation energy)'),
        Patch(facecolor='#2ECC71', label='z_ΔH (Enthalpy)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')

    # Add reference line at 1.0
    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax.text(1.02, 0.98, 'Amplification = 1.0', transform=ax.get_xaxis_transform(),
           fontsize=9, va='top')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'noise_amplification.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved noise amplification plot to {save_path}")


def generate_report(propagation_results, amplification_results, output_dir):
    """Generate comprehensive report on measurement noise effects."""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("MEASUREMENT NOISE EFFECTS ON CHEMICAL PARAMETERS")
    report_lines.append("=" * 80)
    report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("\n" + "-" * 80)

    # Section 1: Overview
    report_lines.append("\n1. ANALYSIS OVERVIEW")
    report_lines.append("-" * 40)
    report_lines.append("""
This analysis examines how measurement errors in training data propagate to
affect the learned chemical kinetic parameters. The pathway is:

  Measurement Error → Data Loss trains NN → NN predictions (C) are biased
  → PDE residual uses biased C → Chemical parameters receive biased gradients

Key insight: Even though chemical parameters don't receive gradients directly
from the data loss, they ARE affected indirectly through the PDE loss computation.
""")

    # Section 2: Gradient Propagation
    report_lines.append("\n" + "-" * 80)
    report_lines.append("\n2. GRADIENT PROPAGATION ANALYSIS")
    report_lines.append("-" * 40)

    baseline = propagation_results['baseline']
    report_lines.append(f"\nBaseline (no noise):")
    report_lines.append(f"  PDE Loss: {baseline['pde_loss']:.6f}")
    report_lines.append(f"  Data Loss: {baseline['data_loss']:.6f}")

    report_lines.append("\nEffect of noise on gradients:")
    report_lines.append(f"{'Noise Level':<15} {'PDE Loss Δ':<15} {'Avg Grad Δ (%)':<15}")
    report_lines.append("-" * 45)

    for result in propagation_results['noisy_results']:
        noise_pct = result['noise_level'] * 100
        pde_delta = result['pde_loss'] - baseline['pde_loss']

        avg_change = np.mean([gc['relative_change_pct']
                             for gc in result['gradient_changes'].values()
                             if gc['relative_change_pct'] < float('inf')])

        report_lines.append(f"{noise_pct:>10.1f}%     {pde_delta:>+12.6f}   {avg_change:>12.2f}%")

    # Section 3: Amplification Factors
    report_lines.append("\n" + "-" * 80)
    report_lines.append("\n3. NOISE AMPLIFICATION FACTORS")
    report_lines.append("-" * 40)
    report_lines.append(f"\nNoise level tested: {amplification_results['noise_level']*100:.0f}%")
    report_lines.append(f"Number of trials: {amplification_results['n_trials']}")
    report_lines.append("\nAmplification Factor = (% change in gradient) / (% noise in data)")
    report_lines.append("Factor > 1: Noise is amplified")
    report_lines.append("Factor < 1: Noise is dampened")

    # Sort and display top amplifiers
    sorted_factors = sorted(amplification_results['amplification_factors'].items(),
                           key=lambda x: x[1]['amplification_factor'], reverse=True)

    report_lines.append(f"\nTop 10 most sensitive parameters:")
    report_lines.append(f"{'Parameter':<25} {'Amplification':>15}")
    report_lines.append("-" * 40)

    for name, data in sorted_factors[:10]:
        short_name = name.split('.')[-1] if '.' in name else name
        report_lines.append(f"{short_name:<25} {data['amplification_factor']:>15.2f}")

    # Section 4: Implications
    report_lines.append("\n" + "-" * 80)
    report_lines.append("\n4. IMPLICATIONS FOR MODEL RELIABILITY")
    report_lines.append("-" * 40)

    # Calculate summary statistics
    all_factors = [d['amplification_factor'] for d in amplification_results['amplification_factors'].values()]
    avg_amplification = np.mean(all_factors)
    max_amplification = np.max(all_factors)

    report_lines.append(f"""
Summary Statistics:
  - Average amplification factor: {avg_amplification:.2f}
  - Maximum amplification factor: {max_amplification:.2f}

Interpretation:
  - Measurement errors in training data DO affect chemical parameters indirectly
  - The effect is {"amplified" if avg_amplification > 1 else "dampened"} on average
  - Most sensitive parameter: {sorted_factors[0][0].split('.')[-1]}

Recommendations:
  1. Minimize measurement errors in output concentration measurements
  2. Use measurement uncertainty for data weighting if available
  3. Consider ensemble training for uncertainty quantification
  4. Validate learned kinetic parameters against literature values where possible
""")

    report_lines.append("\n" + "=" * 80)

    # Save report
    report_content = "\n".join(report_lines)
    report_path = os.path.join(output_dir, 'measurement_noise_report.txt')
    with open(report_path, 'w') as f:
        f.write(report_content)
    logger.info(f"Saved measurement noise report to {report_path}")

    # Save JSON results
    json_results = {
        'gradient_propagation': propagation_results,
        'amplification_factors': amplification_results
    }

    json_path = os.path.join(output_dir, 'measurement_noise_results.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)
    logger.info(f"Saved measurement noise JSON to {json_path}")

    return report_content
