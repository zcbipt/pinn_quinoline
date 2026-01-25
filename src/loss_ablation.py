# ------------------------------------------------------------------------------------------
# Description: ABLATION STUDY loss functions - no time/feature encoding or fusion layers.
#              Raw inputs are directly fed into the main neural network.
# ------------------------------------------------------------------------------------------

import torch
from loguru import logger
import os
import sys
import numpy as np


ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(ROOT_DIR)
MODEL_DIR = os.path.join(ROOT_DIR, "model")


def compute_pde_loss(model, c_initial=None, time_points=None, env_pred=None, debug=False):
    """
    ABLATION: Compute the PDE residual using raw inputs directly (no encoding/fusion).

    Args:
        model: PINN model (ablation version without encoders)
        c_initial: Initial concentrations [batch_size, n_features]
        time_points: Tensor of time points [n_time_points]
        env_pred: Predicted environment values [batch_size, n_time_points, n_env_vars]
        debug: Whether to print debug information

    Returns:
        Tuple of residuals for each species
    """
    device = model.device
    batch_size = c_initial.shape[0]
    n_time_points = len(time_points)

    def debug_parameter_version(name, param):
        if hasattr(param, '_version'):
            logger.debug(f"Parameter '{name}' shape={param.shape}, version={param._version}, "
                         f"requires_grad={param.requires_grad}")
        return param

    # -----Constants-----
    A_TRANSFORM_BASE = 10.0
    E_TRANSFORM_MULTIPLIER = 79000.0
    E_TRANSFORM_OFFSET = 1000.0
    R = 8.314
    r_inv = 1.0 / R

    # -----Initialize residual collections-----
    all_residuals = [[] for _ in range(7)]

    # -----ABLATION: Extract raw sample features (no encoding)-----
    sample_features = torch.cat([
        c_initial[:, 1:2],
        c_initial[:, 2:3],
        c_initial[:, 3:11],
        c_initial[:, 11:14],
    ], dim=1)

    # -----Pre-compute helper functions for rate and equilibrium constants-----
    def compute_rate_constant(A_param, E_param, T):
        exp_arg = -(E_TRANSFORM_OFFSET + E_TRANSFORM_MULTIPLIER * torch.sigmoid(E_param)) * r_inv / T
        return A_TRANSFORM_BASE ** (10.0 * A_param) * torch.exp(exp_arg)

    def compute_equil_constant(A_param, H_param, T):
        exp_arg = (E_TRANSFORM_OFFSET + E_TRANSFORM_MULTIPLIER * torch.sigmoid(H_param)) * r_inv / T
        return A_TRANSFORM_BASE ** (10.0 * A_param) * torch.exp(exp_arg)

    # -----Process time points-----
    for t_idx in range(n_time_points):
        t_tensor = torch.ones(batch_size, 1, dtype=torch.float64, device=device) * time_points[t_idx]
        t_tensor.requires_grad_(True)

        if model.pinn_inputs == 'time':
            c_pred_batch = model.net(t_tensor)
        elif model.pinn_inputs == 'time+initials':
            combined_features = torch.cat([t_tensor, sample_features], dim=1)
            c_pred_batch = model.net(combined_features)
        else:
            raise ValueError(f"Invalid PINN inputs: {model.pinn_inputs}")

        CA_out, CB_out, CC_out, CD_out, CE_out, CF_out, CG_out, CI_out = [c_pred_batch[:, i] for i in range(8)]

        all_grads = []
        for species_out in [CA_out, CB_out, CC_out, CD_out, CE_out, CF_out, CG_out]:
            batch_ones = torch.ones_like(species_out)
            grad = torch.autograd.grad(
                outputs=species_out,
                inputs=t_tensor,
                grad_outputs=batch_ones,
                create_graph=True,
                retain_graph=True
            )[0]
            all_grads.append(grad.squeeze(1))

        dCA_dt, dCB_dt, dCC_dt, dCD_dt, dCE_dt, dCF_dt, dCG_dt = all_grads

        T_batch = env_pred[:, t_idx, 0]
        pH_batch = env_pred[:, t_idx, 4]

        k_values = []
        for idx in range(1, 11):
            A_param = getattr(model, f'x_A_{idx}')
            E_param = getattr(model, f'y_E_{idx}')
            k_values.append(compute_rate_constant(A_param, E_param, T_batch))

        k_1, k_2, k_3, k_4, k_5, k_6, k_7, k_8, k_9, k_10 = k_values

        K_values = []
        for s in ['H', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'I']:
            A_param = getattr(model, f'x_A_{s}')
            H_param = getattr(model, f'z_Delta_H_{s}')
            K_values.append(compute_equil_constant(A_param, H_param, T_batch))

        K_H, K_A, K_B, K_C, K_D, K_E, K_F, K_G, K_I = K_values

        ADS = 1.0 + K_H * pH_batch + \
            K_A * CA_out + K_B * CB_out + K_C * CC_out + K_D * CD_out + \
            K_E * CE_out + K_F * CF_out + K_G * (torch.zeros_like(CG_out) + 1e-10) + K_I * CI_out

        term_KH_pH = K_H * pH_batch
        term_KA_CA = K_A * CA_out
        term_KB_CB = K_B * CB_out
        term_KC_CC = K_C * CC_out

        reaction_term_A = (-k_1 * term_KH_pH * term_KA_CA - k_4 * term_KH_pH * term_KA_CA) / ADS
        reaction_term_B = (k_1 * term_KH_pH * term_KA_CA - k_2 * term_KH_pH * term_KB_CB - k_5 * term_KH_pH * term_KB_CB) / ADS
        reaction_term_C = (k_2 * term_KH_pH * term_KB_CB - k_3 * term_KH_pH * term_KC_CC - k_6 * term_KH_pH * term_KC_CC) / ADS
        reaction_term_D = (k_3 * term_KH_pH * term_KC_CC - k_7 * term_KH_pH * K_D * CD_out) / ADS
        reaction_term_E = (k_4 * term_KH_pH * term_KA_CA - k_8 * term_KH_pH * K_E * CE_out) / ADS
        reaction_term_F = (k_5 * term_KH_pH * term_KB_CB + k_8 * term_KH_pH * K_E * CE_out - k_9 * term_KH_pH * K_F * CF_out) / ADS
        reaction_term_G = (k_6 * term_KH_pH * term_KC_CC + k_9 * term_KH_pH * K_F * CF_out - k_10 * term_KH_pH * K_G * (torch.zeros_like(CG_out) + 1e-10)) / ADS

        residuals = [
            dCA_dt - reaction_term_A,
            dCB_dt - reaction_term_B,
            dCC_dt - reaction_term_C,
            dCD_dt - reaction_term_D,
            dCE_dt - reaction_term_E,
            dCF_dt - reaction_term_F,
            dCG_dt - reaction_term_G
        ]

        for i in range(len(residuals)):
            all_residuals[i].append(residuals[i])

        if (t_idx + 1) % 5 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -----Concatenate all residuals-----
    final_residuals = [torch.cat(res_list) for res_list in all_residuals]

    return tuple(final_residuals)


def hybrid_concentration_loss(pred, target, epsilon=1e-6, weight_relative=0.6, return_per_sample=False):
    """
    Robust hybrid loss optimized for multi-scale chemical concentration data

    Args:
        pred: Predicted values
        target: Target values
        epsilon: Small value for numerical stability
        weight_relative: Weight for relative error component (0-1)

    Returns:
        Weighted combination of relative and log-space errors
    """
    # 1. Ensure positive values with proper clamping
    pred_safe = torch.clamp(pred, min=epsilon)
    target_safe = torch.clamp(target, min=epsilon)

    # Calculate relative error per sample
    rel_error = torch.abs(pred_safe - target_safe) / (target_safe + epsilon)

    # 3. Calculate log-space error to handle order-of-magnitude differences
    log_pred = torch.log10(pred_safe)
    log_target = torch.log10(target_safe)
    log_error = torch.abs(log_pred - log_target)

    # 4. Combine both errors with proper weighting
    combined_error = weight_relative * rel_error + (1 - weight_relative) * log_error

    # Get per-sample losses
    per_sample_loss = torch.mean(combined_error, dim=-1)  # Average over species dimension only

    if return_per_sample:
        return per_sample_loss

    # Average over samples
    return torch.mean(per_sample_loss)


def r2_score(y_true, y_pred):
    """
    Calculate the R² score (coefficient of determination) between true and predicted values

    Args:
        y_true: Tensor or numpy array of true values
        y_pred: Tensor or numpy array of predicted values

    Returns:
        R² score as a scalar
    """
    # Check if inputs are numpy arrays
    if isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray):
        # NumPy implementation
        total_sum_squares = np.sum((y_true - np.mean(y_true))**2)
        residual_sum_squares = np.sum((y_true - y_pred)**2)
        r2 = 1 - (residual_sum_squares / total_sum_squares)
        return r2
    else:
        # PyTorch implementation
        total_sum_squares = torch.sum((y_true - torch.mean(y_true))**2)
        residual_sum_squares = torch.sum((y_true - y_pred)**2)
        r2 = 1 - (residual_sum_squares / total_sum_squares)
        return r2.item() if hasattr(r2, 'item') else r2


def compute_mass_conservation_penalty(c_pred=None, c_initial=None):
    """
    Compute mass conservation penalty using MSE.

    Args:
        c_pred: Predicted concentrations [batch_size, time_steps, n_species]
        c_initial: Initial concentrations [batch_size, n_species]

    Returns:
        Mass conservation penalty using MSE
    """
    # Calculate true total mass from initial conditions
    true_total_mass = torch.sum(c_initial, dim=1, keepdim=True)  # [batch_size, 1]

    # Total mass at each timestep (sum across species dimension)
    predicted_total_mass = torch.sum(c_pred, dim=2)  # [batch_size, time_steps]

    # Compute MSE between each timestep's total mass and the true initial mass
    mass_mse = torch.mean((predicted_total_mass - true_total_mass)**2)

    return mass_mse
