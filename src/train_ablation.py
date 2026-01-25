# ------------------------------------------------------------------------------------------
# Description: ABLATION STUDY training module - uses loss_ablation.py for raw input model.
# ------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
from loguru import logger
from pathlib import Path
import os
import time
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd


ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(ROOT_DIR)
MODEL_DIR = os.path.join(ROOT_DIR, "model")

# ABLATION: Import from loss_ablation instead of loss
from .loss_ablation import compute_pde_loss, compute_mass_conservation_penalty


def toggle_parameter_freezing(model, epoch, phase_transition_epoch=300, frozen_parameters=None):
    """Freeze/unfreeze parameters based on training phase
    
    In Phase 1 (epoch < phase_transition_epoch): Freeze dynamics parameters
    In Phase 2 (epoch >= phase_transition_epoch): Unfreeze all parameters
    """
    if frozen_parameters is None:
        frozen_params = ['y_E_', 'x_A_', 'z_Delta_H_']
    else:
        frozen_params = frozen_parameters
        
    if epoch < phase_transition_epoch:
        for name, param in model.named_parameters():
            if any(x in name for x in frozen_params):
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        if epoch == 0 or epoch % 50 == 0:  # Log only occasionally
            logger.info(f"Phase 1 (epoch {epoch}): Training initial condition parameters only")
    
    elif epoch == phase_transition_epoch:
        # Transition to Phase 2: Unfreeze all parameters
        for name, param in model.named_parameters():
            param.requires_grad = True
        logger.info(f"Phase 2 (epoch {epoch}): Unfreezing all parameters for dynamics learning")


def train_pinn(model=None, n_epochs=None, input_features=None, output_targets=None,
               checkpoint_dir=None, model_folder=None, checkpoint_freq=1000, run_mode=None, pinn_inputs=None, resume_from=None,
               c_scaling_factor=1e5, task=None, logging_freq=100, adaptive_start_epoch=None, debug=False, curriculum_config=None):
    """
    Train a Physics-Informed Neural Network with regular checkpointing
    
    Args:
        model: PINN model instance
        n_epochs: Total number of epochs
        input_features: Input features for data loss
        output_targets: Target values for data loss
        checkpoint_dir: Directory to save checkpoints (default: "checkpoints")
        checkpoint_freq: Save checkpoint every N epochs (default: 500)
        run_mode: Mode to run the model ('pinn' or 'pinn+phase')
        pinn_inputs: Inputs to the PINN (['time', 'time+initials'])
        resume_from: Path to checkpoint to resume from (optional)
        debug: Whether to print debug information
        
    Returns:
        parameters_history: History of model parameters
        loss_history: History of loss values
    """
    assert pinn_inputs in ['time', 'time+initials'], f"Invalid PINN inputs: {pinn_inputs}"
    assert run_mode in ['pinn', 'pinn+phase'], f"Invalid run mode: {run_mode}"
    assert task in ['train', 'predict'], f"Invalid task: {task}. Must be 'train' or 'predict'."

    logging_freq = logging_freq
    
    # -----Enable data shuffling for each epoch-----
    enable_data_shuffling = True  # Toggle for data shuffling

    # -----Set scaling factor of C* concentrations-----
    c_scaling_factor = c_scaling_factor

    if debug:
        torch.autograd.set_detect_anomaly(True)
    else:
        torch.autograd.set_detect_anomaly(False)

    def debug_tensor_version(name, tensor):
        if hasattr(tensor, '_version'):
            logger.debug(f"Tensor '{name}' shape={tensor.shape}, version={tensor._version}, "
                         f"requires_grad={tensor.requires_grad}, is_leaf={tensor.is_leaf}")
        return tensor

    # -----Get device from model-----
    device = model.device

    # -----Move input data to device-----
    input_features = input_features.to(device, dtype=torch.float64)
    output_targets = output_targets.to(device, dtype=torch.float64)

    # -----Setup checkpoint directory-----
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(MODEL_DIR, "default_checkpoints")
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")

    # -----Set up training configuration based on input mode-----
    include_initial_loss = True

    # -----Initial weights-----
    if curriculum_config is not None and 'initial_weights' in curriculum_config:
        if task == 'predict' and 'predict' in curriculum_config['initial_weights']:
            weights = curriculum_config['initial_weights']['predict'].copy()
        elif task == 'train' and 'train' in curriculum_config['initial_weights']:
            weights = curriculum_config['initial_weights']['train'].copy()
        else:
            if task == 'predict':
                weights = {
                    'pde': 1e0,
                    'initial': 1e0,
                    'mass': 1e0,
                }
            else:
                weights = {
                    'data': 1e0,
                    'pde': 1e0,
                    'initial': 1e0,
                    'mass': 1e0,
                }
    else:
        if task == 'predict':
            weights = {
                'pde': 1e0,
                'initial': 1e0,
                'mass': 1e0,
            }
        else:
            weights = {
                'data': 1e0,
                'pde': 1e0,
                'initial': 1e0,
                'mass': 1e0,
            }

    if run_mode == 'pinn+phase':
        weights['phase'] = 0.0
    if task == 'predict':
        if 'data' in weights:
            del weights['data']

    # -----Curriculum Learning Stages-----
    if curriculum_config is not None and 'stages' in curriculum_config:
        curriculum_stages = curriculum_config['stages']
        curriculum_stages = {
            int(k): {
                loss_type: float(loss_value) if isinstance(loss_value, (str, int, float)) else loss_value
                for loss_type, loss_value in v.items()
            } 
            for k, v in curriculum_stages.items()
        }
    else:
        if task == 'predict':
            curriculum_stages = {}
        else:
            curriculum_stages = {
                0: {'data': 1e2, 'pde': 1e-15, 'initial': 1e3, 'mass': 1e1},
                200: {'data': 1e2, 'pde': 2e-14, 'initial': 3e2, 'mass': 1e1},
                700: {'data': 1e2, 'pde': 1e-12, 'initial': 8e1, 'mass': 1e1},
            }

    # -----Define when to switch from curriculum to adaptive-----
    curriculum_end_epoch = adaptive_start_epoch

    # -----GradNorm hyperparameters-----
    gradnorm_alpha = curriculum_config.get('gradnorm_alpha', 0.01) if curriculum_config else 0.01
    gradnorm_lr = curriculum_config.get('gradnorm_lr', 0.001) if curriculum_config else 0.001
    
    # -----Target relative training rates for different tasks-----
    if curriculum_config is not None and 'target_rates' in curriculum_config:
        target_rates = curriculum_config['target_rates'].copy()
    else:
        target_rates = {
            'data': 1.0,
            'pde': 1.0,
            'initial': 1.0,
            'mass': 1.0,
            'phase': 1.0,
        }

    # -----Weight adjustment frequency-----
    if curriculum_config is not None and 'weight_update_frequency' in curriculum_config:
        weight_update_freq = curriculum_config['weight_update_frequency']
    else:
        weight_update_freq = 200

    # -----Store gradient history for visualization-----
    grad_history = {k: [] for k in weights.keys()}
    weight_history = {k: [] for k in weights.keys()}
    
    # -----Store loss history for GradNorm-----
    loss_history_gradnorm = {k: [] for k in weights.keys()}

    # -----Caps on weight values to prevent extreme values-----
    if curriculum_config is not None and 'weight_caps' in curriculum_config:
        if task == 'predict' and 'predict' in curriculum_config['weight_caps']:
            weight_caps = {k: tuple(float(val) for val in v) for k, v in curriculum_config['weight_caps']['predict'].items()}
        elif task == 'train' and 'train' in curriculum_config['weight_caps']:
            weight_caps = {k: tuple(float(val) for val in v) for k, v in curriculum_config['weight_caps']['train'].items()}
        else:
            if task == 'predict':
                weight_caps = {
                    'pde': (1e-3, 1e-3),
                    'initial': (1e-3, 1e3),
                    'mass': (1e-3, 1e3),
                }
            else:
                weight_caps = {
                    'data': (1e-3, 1e3),
                    'pde': (1e-3, 1e3),
                    'initial': (1e-3, 1e3),
                    'mass': (1e-3, 1e3),
                }
    else:
        if task == 'predict':
            weight_caps = {
                'pde': (1e-3, 1e3),
                'initial': (1e-3, 1e3),
                'mass': (1e-3, 1e3),
            }
        else:
            weight_caps = {
                'data': (1e-3, 1e3),
                'pde': (1e-3, 1e3),
                'initial': (1e-3, 1e3),
                'mass': (1e-3, 1e3),
            }
    
    if run_mode == 'pinn+phase':
        if (curriculum_config is not None and 'weight_caps' in curriculum_config and 
            task in curriculum_config['weight_caps'] and 
            'phase' in curriculum_config['weight_caps'][task]):
            pass
        else:
            weight_caps['phase'] = (1e-3, 1e3)

    # -----Phase loss weight schedule - more gradual introduction-----
    phase_intro_epoch = curriculum_config.get('phase_loss_introduction_epoch', 3000) if curriculum_config else 3000

    # -----Add weight explosion detection and rollback-----
    def detect_weight_explosion(current_weights, previous_weights, loss_value, threshold_factor=15.0, loss_threshold=1e5):
        """
        Detect if weights have exploded based on:
        1. Sudden weight increases
        2. Loss value explosion
        """
        if not previous_weights:
            return False, "No previous weights to compare"
        
        for key in current_weights:
            if key in previous_weights:
                ratio = current_weights[key] / max(previous_weights[key], 1e-10)
                if ratio > threshold_factor:
                    return True, f"Weight '{key}' increased {ratio:.2f}x (from {previous_weights[key]:.2e} to {current_weights[key]:.2e})"
        
        if loss_value > loss_threshold:
            return True, f"Loss exploded to {loss_value:.2e} (threshold: {loss_threshold:.2e})"
        
        return False, "Weights stable"

    # -----Learning rate schedule with warmup-----
    lr_schedule = {}
    
    if curriculum_config is not None and 'learning_rate' in curriculum_config:
        lr_config = curriculum_config['learning_rate']
        warmup_epochs = lr_config.get('warmup_epochs', 20)
        base_lr = lr_config.get('base_lr', 1e-4)
        min_lr = lr_config.get('min_lr', 1e-5)
        
        for i in range(warmup_epochs + 1):
            lr_schedule[i] = min_lr + (base_lr - min_lr) * (i / warmup_epochs)
        
        if 'schedule' in lr_config:
            for epoch_str, lr_value in lr_config['schedule'].items():
                lr_schedule[int(epoch_str)] = float(lr_value)
        else:
            lr_schedule[warmup_epochs] = 1e-4
            lr_schedule[20000] = 1e-5
    else:
        warmup_epochs = 20
        base_lr = 1e-4
        min_lr = 1e-5
        
        for i in range(warmup_epochs + 1):
            lr_schedule[i] = min_lr + (base_lr - min_lr) * (i / warmup_epochs)
        
        lr_schedule[warmup_epochs] = 1e-4
        lr_schedule[20000] = 1e-5

    # -----Initialize tracking variables-----
    start_epoch = 0
    best_loss = float('inf')
    
    # -----Initialize optimizer-----
    thermodynamic_params = []
    other_params = []

    for name, param in model.named_parameters():
        if 'z_Delta_H_' in name or 'y_E_' in name or 'x_A_' in name:
            thermodynamic_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_schedule[0])
    
    lbfgs_switch_fraction = curriculum_config.get('lbfgs_switch_fraction', 0.8) if curriculum_config else 0.8
    lbfgs_switch_epoch = start_epoch + int(lbfgs_switch_fraction * n_epochs)  # Switch at specified fraction of training
    lbfgs_optimizer = None

    mse_loss = nn.MSELoss()
    parameters_history = []
    loss_history = []

    # -----For tracking losses and weight adjustments-----
    adaptive_weight_log = []

    # -----Store previous weights for stability checks-----
    previous_weights = weights.copy()
    stable_weights_backup = weights.copy()

    # -----Resume from checkpoint if provided-----
    if resume_from is not None and os.path.exists(resume_from):
        logger.info(f"Loading checkpoint from: {resume_from}")

        map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(resume_from, map_location=map_location, weights_only=False)

        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

        # -----Load optimizer state for continuity (momentum, etc.)-----
        if 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info("✅ Loaded optimizer state (momentum, learning rate history)")
            except Exception as e:
                logger.warning(f"⚠️ Could not load optimizer state: {e}")
                logger.warning("Optimizer state will be reset (momentum, etc.)")

        best_loss = checkpoint['best_loss']
        
        # -----Set learning rate based on mode-----
        if task == 'predict':
            predict_lr = max(1e-5, min(5e-5, lr_schedule[max([e for e in lr_schedule.keys() if e <= start_epoch])]))
            logger.info(f"🔄 Predict mode: Using learning rate {predict_lr:.2e} (checkpoint would use {lr_schedule[max([e for e in lr_schedule.keys() if e <= start_epoch])]:.2e})")
            for param_group in optimizer.param_groups:
                param_group['lr'] = predict_lr
        else:
            current_lr = lr_schedule[max([e for e in lr_schedule.keys() if e <= start_epoch])]
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        
        # -----Add existing history if available-----
        if 'parameters_history' in checkpoint:
            parameters_history = checkpoint['parameters_history']
        if 'loss_history' in checkpoint:
            loss_history = checkpoint['loss_history']

        # -----Load GradNorm loss history if available-----
        if 'loss_history_gradnorm' in checkpoint:
            checkpoint_gradnorm_history = checkpoint['loss_history_gradnorm']
            for k in weights.keys():
                if k in checkpoint_gradnorm_history:
                    loss_history_gradnorm[k] = checkpoint_gradnorm_history[k].copy()
                    logger.info(f"Loaded GradNorm history for '{k}': {len(loss_history_gradnorm[k])} entries")
            logger.info(f"GradNorm loss history loaded for continuous adaptive weighting")
        else:
            logger.warning("No GradNorm loss history found in checkpoint - starting with empty history")

        # -----If resuming with adaptive weights, can also load weight history-----
        if 'adaptive_weights' in checkpoint:
            weights = checkpoint['adaptive_weights']
            logger.info(f"Loaded adaptive weights from checkpoint: {weights}")
            
            # -----Re-initialize history structures if weights changed-----
            for k in weights.keys():
                if k not in loss_history_gradnorm:
                    loss_history_gradnorm[k] = []
                    logger.info(f"Initialized empty GradNorm history for new weight key: {k}")
            
            for k in weights.keys():
                if k not in grad_history:
                    grad_history[k] = []
                if k not in weight_history:
                    weight_history[k] = []
            
            for history_dict, name in [(loss_history_gradnorm, "GradNorm"), (grad_history, "gradient"), (weight_history, "weight")]:
                keys_to_remove = [k for k in history_dict.keys() if k not in weights]
                for k in keys_to_remove:
                    del history_dict[k]
                    logger.info(f"Removed {name} history for unused weight key: {k}")
            
        logger.info(f"Resuming training from epoch {start_epoch} with best loss: {best_loss:.4e}")
    
    def save_checkpoint(epoch, loss, curr_weights=None, is_best=False, save_periodic=False, save_final=False):
        """Save model checkpoint with training state"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'best_loss': best_loss,
            'parameters_history': parameters_history,
            'loss_history': loss_history,
            'loss_history_gradnorm': loss_history_gradnorm,
            'device': str(model.device),
            'adaptive_weights': curr_weights
        }
        
        # -----Save periodic checkpoint-----
        if save_periodic:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}_{timestamp}.pt')
            torch.save(checkpoint, checkpoint_path)
        
        # -----Save final checkpoint-----
        if save_final:
            final_path = os.path.join(checkpoint_dir, f'checkpoint_final_{epoch}_{timestamp}.pt')
            torch.save(checkpoint, final_path)
        
        # -----Always save as latest checkpoint (overwrite) for any save operation-----
        if save_periodic or save_final or is_best:
            latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pt')
            torch.save(checkpoint, latest_path)
        
        # -----Save best checkpoint-----
        if is_best:
            best_path = os.path.join(checkpoint_dir, 'checkpoint_best.pt')
            torch.save(checkpoint, best_path)
    
    # -----Preprocess initial and final inputs-----
    c_initial, env_dict = model.preprocess_initial_inputs(x=input_features)
    c_final, env_dict_final = model.preprocess_initial_inputs(x=output_targets)

    env_dict_cpu = {k: v.cpu().detach().numpy() for k, v in env_dict.items()}
    env_dict_df = pd.DataFrame(env_dict_cpu)
    env_dict_df.to_csv(os.path.join(model_folder, 'env_dict.csv'), index=False)

    env_dict_final_cpu = {k: v.cpu().detach().numpy() for k, v in env_dict_final.items()}
    env_dict_final_df = pd.DataFrame(env_dict_final_cpu)
    env_dict_final_df.to_csv(os.path.join(model_folder, 'env_dict_final.csv'), index=False)

    # -----Replace C* concentrations with XF* concentrations-----
    c_initial = torch.cat([
        c_initial[:, :3],
        input_features[:, 5:13] * c_scaling_factor,
        c_initial[:, 11:]
    ], dim=1)

    c_final = torch.cat([
        c_final[:, :3],
        output_targets[:, 5:13] * c_scaling_factor,
        c_final[:, 11:]
    ], dim=1)

    if debug:
        logger.debug("=== Input Diversity Check ===")
        for sample_idx in range(min(5, c_initial.shape[0])):
            logger.debug(f"\nSample {sample_idx} features:")
            logger.debug(f"Time: {c_initial[sample_idx, 0].item():.4f}")
            logger.debug(f"Temperature: {c_initial[sample_idx, 1].item():.4f}")
            logger.debug(f"Initial concentrations: {c_initial[sample_idx, 3:11].detach().cpu().numpy()}")
            logger.debug("=== Output Diversity Check ===")
        for sample_idx in range(min(5, c_final.shape[0])):
            logger.debug(f"\nSample {sample_idx} features:")
            logger.debug(f"Time: {c_final[sample_idx, 0].item():.4f}")
            logger.debug(f"Temperature: {c_final[sample_idx, 1].item():.4f}")
            logger.debug(f"Final concentrations: {c_final[sample_idx, 3:11].detach().cpu().numpy()}")

    # -----Sanity check for CA concentration decrease-----
    ca_initial = c_initial[:, 3:11][:, 0]
    ca_final = c_final[:, 3:11][:, 0]

    ca_decrease = ca_initial > ca_final

    # Log statistics about the filtering
    total_samples = len(ca_initial)
    valid_samples = ca_decrease.sum().item()
    logger.info(f"CA concentration check - Total samples: {total_samples}, Valid samples: {valid_samples}")
    logger.warning(f"Removed {total_samples - valid_samples} samples where CA didn't decrease")

    for i in range(total_samples):
        if not ca_decrease[i]:
            logger.warning(f"Sample {i}: CA increased from {ca_initial[i]:.6f} to {ca_final[i]:.6f} "
                        f"(change: {(ca_final[i] - ca_initial[i]):.6f})")

    valid_mask = ca_decrease
    c_initial = c_initial[valid_mask]
    c_final = c_final[valid_mask]

    env_dict = {k: v[valid_mask] for k, v in env_dict.items()}
    env_dict_final = {k: v[valid_mask] for k, v in env_dict_final.items()}

    if len(c_initial) == 0:
        raise ValueError("No valid samples remaining after CA concentration check!")

    # -----Make sure c_initial, c_final, env_dict, env_dict_final have the right dtype-----
    c_initial = c_initial.to(device, dtype=torch.float64)
    env_dict = {k: v.to(device, dtype=torch.float64) for k, v in env_dict.items()}
    c_final = c_final.to(device, dtype=torch.float64)
    env_dict_final = {k: v.to(device, dtype=torch.float64) for k, v in env_dict_final.items()}

    # -----Pre-training initialization-----
    if start_epoch == 0 and task == 'train':  # Only for fresh training, not when resuming or predict
        logger.info("Pre-training initialization: Focusing on initial condition matching")
        ic_params = []
        for name, param in model.named_parameters():
            if not any(x in name for x in ['y_E_', 'x_A_', 'z_Delta_H_']):
                ic_params.append(param)
        
        pre_optimizer = torch.optim.Adam(ic_params, lr=5e-3)
        
        for pre_epoch in range(300):
            if enable_data_shuffling:
                pre_shuffle_indices = torch.randperm(c_initial.shape[0], device=device)
                pre_c_initial = c_initial[pre_shuffle_indices]
            else:
                pre_c_initial = c_initial
                
            pre_optimizer.zero_grad()
            
            with torch.enable_grad():
                c_pred, _ = model(c_initial=pre_c_initial)
                
                ic_loss = mse_loss(c_pred[:, 0, :], pre_c_initial[:, 3:11])
                
                ic_loss.backward(retain_graph=True)
                pre_optimizer.step()
            
            if pre_epoch % 10 == 0:
                logger.info(f"Pre-training epoch {pre_epoch}: Initial condition loss: {ic_loss.item():.4e}")
        
        logger.info("Pre-training complete. Starting main training loop.")

    # -----Main training loop-----
    for epoch in range(start_epoch, start_epoch + n_epochs):
        # -----Shuffle data for each epoch-----
        if enable_data_shuffling:
            shuffle_indices = torch.randperm(c_initial.shape[0], device=device)
            
            c_initial_epoch = c_initial[shuffle_indices]
            c_final_epoch = c_final[shuffle_indices]
            env_dict_epoch = {k: v[shuffle_indices] for k, v in env_dict.items()}
            env_dict_final_epoch = {k: v[shuffle_indices] for k, v in env_dict_final.items()}
            
            if debug:
                logger.debug(f"Epoch {epoch}: Using shuffled data with {len(c_initial_epoch)} samples")
        else:
            c_initial_epoch = c_initial
            c_final_epoch = c_final
            env_dict_epoch = env_dict
            env_dict_final_epoch = env_dict_final
        
        # -----For 'predict' task, allow toggling training of all parameters including dynamics-----
        if task == 'predict':
            for name, param in model.named_parameters():
                if any(x in name for x in ['y_E_', 'x_A_', 'z_Delta_H_']):
                    param.requires_grad = True
                else:
                    param.requires_grad = True
            
            if epoch == 0 or epoch % logging_freq == 0:
                logger.info(f"Predict mode: Training all parameters including dynamics")
        elif task == 'train':
            phase_transition_epoch = curriculum_config.get('parameter_freezing', {}).get('phase_transition_epoch', 3000) if curriculum_config else 3000
            frozen_parameters = curriculum_config.get('parameter_freezing', {}).get('frozen_parameters', None) if curriculum_config else None
            toggle_parameter_freezing(model, epoch, phase_transition_epoch=phase_transition_epoch, frozen_parameters=frozen_parameters)

        # -----Determine weight update strategy with gradual transition-----
        transition_buffer = min(200, curriculum_end_epoch // 2)  # Adaptive transition period
        transition_start = max(0, curriculum_end_epoch - transition_buffer)
        
        if epoch == transition_start and transition_start > 0:
            logger.warning(f"🚀 ENTERING TRANSITION PHASE at epoch {epoch} (until epoch {curriculum_end_epoch})")
            logger.warning(f"Current weights: {weights}")
        elif epoch == curriculum_end_epoch:
            logger.warning(f"🎯 STARTING FULL ADAPTIVE PHASE at epoch {epoch}")
            logger.warning(f"Loss history lengths: {[len(loss_history_gradnorm[k]) for k in loss_history_gradnorm.keys()]}")
            logger.warning(f"Starting weights: {weights}")
        
        if epoch < transition_start and curriculum_stages:
            current_stage = max([stage for stage in curriculum_stages.keys() if stage <= epoch])
            target_weights = curriculum_stages[current_stage]
            
            next_stages = [stage for stage in curriculum_stages.keys() if stage > current_stage]
            if next_stages:
                next_stage = min(next_stages)
                next_weights = curriculum_stages[next_stage]
                alpha = (epoch - current_stage) / (next_stage - current_stage)
                for k in weights.keys():
                    if k in target_weights and k in next_weights:
                        weights[k] = (1 - alpha) * target_weights[k] + alpha * next_weights[k]
            else:
                for k in weights.keys():
                    if k in target_weights:
                        weights[k] = target_weights[k]
                        
        elif epoch < curriculum_end_epoch and curriculum_stages:
            current_stage = max([stage for stage in curriculum_stages.keys() if stage <= epoch])
            target_weights = curriculum_stages[current_stage]
            
            for k in weights.keys():
                if k in target_weights:
                    weights[k] = target_weights[k]
            
            if epoch % 10 == 0:
                loss_terms = {}
                with torch.enable_grad():
                    c_pred, time_points = model(c_initial=c_initial_epoch)
                    x_pred, env_pred = model.process_c_pred(c_pred=c_pred, env_dict=env_dict_epoch, time_points=time_points)
                    c_transformed = model.get_phase_tranformed_c(x_pred=x_pred, time_points=time_points, env_pred=env_pred)
                    u_pred, c_pred_at_target = model.get_predictions(x_pred=x_pred, c_pred=c_pred, c_initial=c_initial_epoch, 
                                                                     time_points=time_points, env_pred=env_pred)
                    
                    if task == 'predict':
                        loss_data = torch.tensor(0.0, device=device, dtype=torch.float64)
                    elif run_mode == 'pinn+phase':
                        if enable_data_shuffling:
                            loss_data = mse_loss(u_pred[:, 5:13], output_targets[valid_mask][shuffle_indices][:, 5:13])
                        else:
                            loss_data = mse_loss(u_pred[:, 5:13], output_targets[valid_mask][:, 5:13])
                    elif run_mode == 'pinn':
                        loss_data = mse_loss(c_pred_at_target[:, :], c_final_epoch[:, 3:11])
                    else:
                        raise ValueError(f"Invalid run_mode: {run_mode}")

                    if include_initial_loss:
                        loss_initial_condition = mse_loss(c_pred[:, 0, :], c_initial_epoch[:, 3:11])
                    else:
                        loss_initial_condition = torch.tensor(0.0, device=device, dtype=torch.float64)
                    
                    r1_col, r2_col, r3_col, r4_col, r5_col, r6_col, r7_col = compute_pde_loss(
                        model, c_initial=c_initial_epoch, time_points=time_points, env_pred=env_pred,
                    )
                    loss_pde_col = sum(torch.mean(r ** 2) for r in [r1_col, r2_col, r3_col, r4_col, r5_col, r6_col, r7_col])
                    
                    mass_conservation = compute_mass_conservation_penalty(c_pred=c_pred, c_initial=c_initial_epoch[:, 3:11])
                
                loss_terms = {
                    'data': loss_data,
                    'pde': loss_pde_col,
                    'initial': loss_initial_condition,
                    'mass': mass_conservation
                }
                
                for k in weights.keys():
                    if k in loss_terms:
                        if torch.is_tensor(loss_terms[k]):
                            loss_history_gradnorm[k].append(loss_terms[k].item())
                        else:
                            loss_history_gradnorm[k].append(loss_terms[k])
                
                logger.info(f"Transition Phase - Building loss history at epoch {epoch}")
                
        # -----Full Adaptive weight updates-----
        elif epoch % weight_update_freq == 0:
            loss_terms = {}
            
            with torch.enable_grad():
                c_pred, time_points = model(c_initial=c_initial_epoch)
                x_pred, env_pred = model.process_c_pred(c_pred=c_pred, env_dict=env_dict_epoch, time_points=time_points)
                c_transformed = model.get_phase_tranformed_c(x_pred=x_pred, time_points=time_points, env_pred=env_pred)
                u_pred, c_pred_at_target = model.get_predictions(x_pred=x_pred, c_pred=c_pred, c_initial=c_initial_epoch, 
                                                                 time_points=time_points, env_pred=env_pred)
                
                if task == 'predict':
                    loss_data = torch.tensor(0.0, device=device, dtype=torch.float64)
                elif run_mode == 'pinn+phase':
                    if enable_data_shuffling:
                        loss_data = mse_loss(u_pred[:, 5:13], output_targets[valid_mask][shuffle_indices][:, 5:13])
                    else:
                        loss_data = mse_loss(u_pred[:, 5:13], output_targets[valid_mask][:, 5:13])
                elif run_mode == 'pinn':
                    loss_data = mse_loss(c_pred_at_target[:, :], c_final_epoch[:, 3:11])
                else:
                    raise ValueError(f"Invalid run_mode: {run_mode}")

                if include_initial_loss:
                    loss_initial_condition = mse_loss(c_pred[:, 0, :], c_initial_epoch[:, 3:11])
                else:
                    loss_initial_condition = torch.tensor(0.0, device=device, dtype=torch.float64)
                
                r1_col, r2_col, r3_col, r4_col, r5_col, r6_col, r7_col = compute_pde_loss(
                    model,
                    c_initial=c_initial_epoch,
                    time_points=time_points,
                    env_pred=env_pred,
                )
                loss_pde_col = sum(torch.mean(r ** 2) for r in [r1_col, r2_col, r3_col, r4_col, r5_col, r6_col, r7_col])
                
                if run_mode == 'pinn+phase':
                    loss_phase_transformed = mse_loss(c_transformed[:, 1:, :], c_pred[:, 1:, :])

                mass_conservation = compute_mass_conservation_penalty(
                    c_pred=c_pred,
                    c_initial=c_initial_epoch[:, 3:11]
                )
            
            loss_terms['data'] = loss_data
            loss_terms['pde'] = loss_pde_col
            loss_terms['initial'] = loss_initial_condition
            loss_terms['mass'] = mass_conservation
            if run_mode == 'pinn+phase' and epoch >= phase_intro_epoch:
                loss_terms['phase'] = loss_phase_transformed
            
            param_grads = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param_grads[name] = param.grad.detach().clone()
            
            grad_magnitudes = {}
            for loss_name in weights.keys():
                if task == 'predict' and loss_name == 'data':
                    continue
                if loss_name not in loss_terms:
                    continue
                
                optimizer.zero_grad()
                
                if torch.is_tensor(loss_terms[loss_name]) and loss_terms[loss_name].requires_grad:
                    loss_terms[loss_name].backward(retain_graph=True)
                
                grad_norm = 0.0
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm += param.grad.norm().item() ** 2
                
                grad_magnitudes[loss_name] = grad_norm ** 0.5
                grad_history[loss_name].append(grad_norm ** 0.5)
            
            if debug:
                logger.debug(f"Gradient magnitudes: {grad_magnitudes}")
                logger.debug(f"Any positive gradients: {any(g > 0 for g in grad_magnitudes.values())}")
                logger.debug(f"All positive gradients: {all(g > 0 for g in grad_magnitudes.values())}")
            
            optimizer.zero_grad()
            for name, param in model.named_parameters():
                if name in param_grads:
                    param.grad = param_grads[name]
            
            # -----Advanced GradNorm-based adaptive weight updates-----
            for k in weights.keys():
                if k in loss_terms:
                    if torch.is_tensor(loss_terms[k]):
                        loss_history_gradnorm[k].append(loss_terms[k].item())
                    else:
                        loss_history_gradnorm[k].append(loss_terms[k])
            
                if all(g > 0 for g in grad_magnitudes.values()):
                    epochs_since_adaptive_start = epoch - curriculum_end_epoch
                    if epochs_since_adaptive_start < 100:
                        conservative_alpha = gradnorm_alpha * 0.3
                        conservative_lr = gradnorm_lr * 0.5
                        logger.info(f"Early adaptive phase - using conservative GradNorm: α={conservative_alpha:.4f}, lr={conservative_lr:.4f}")
                    else:
                        conservative_alpha = gradnorm_alpha
                        conservative_lr = gradnorm_lr
                    
                    new_weights = gradnorm_update_weights(
                        weights=weights,
                        grad_magnitudes=grad_magnitudes,
                        loss_terms=loss_terms,
                        target_rates=target_rates,
                        alpha=conservative_alpha,
                        learning_rate=conservative_lr,
                        loss_history=loss_history_gradnorm,
                        epoch=epoch
                    )
                    
                    for k in new_weights.keys():
                        if k in weight_caps:
                            min_val, max_val = weight_caps[k]
                            min_val, max_val = float(min_val), float(max_val)
                            old_weight = weights[k]
                            
                            if epochs_since_adaptive_start < 100:
                                max_change_factor = 1.2
                                min_change_factor = 0.8
                                new_weights[k] = np.clip(new_weights[k], 
                                                       old_weight * min_change_factor, 
                                                       old_weight * max_change_factor)
                            
                            new_weight_val = float(new_weights[k])
                            if new_weight_val < min_val:
                                new_weights[k] = min_val + 0.1 * (old_weight - min_val)
                            elif new_weight_val > max_val:
                                new_weights[k] = max_val - 0.1 * (max_val - old_weight)
                        
                        if k == 'phase' and epoch < phase_intro_epoch:
                            new_weights[k] = 0.0  # Keep at zero until phase_intro_epoch
                    
                    weights.update(new_weights)
                    
                    weight_explosion_check_needed = True
                    
                    stable_weights_backup = weights.copy()
                    previous_weights = weights.copy()
                    
                    is_exploded = False
                    explosion_msg = None
                    
                    log_entry = {
                        'epoch': epoch,
                        'method': 'GradNorm',
                        'weights': weights.copy(),
                        'grad_magnitudes': grad_magnitudes.copy(),
                        'loss_terms': {k: v.item() if torch.is_tensor(v) else v for k, v in loss_terms.items()},
                        'target_rates': target_rates.copy(),
                        'alpha': gradnorm_alpha,
                        'learning_rate': gradnorm_lr,
                        'exploded': is_exploded,
                        'explosion_msg': explosion_msg,
                        'epochs_since_adaptive_start': epochs_since_adaptive_start,
                        'conservative_alpha': conservative_alpha,
                        'conservative_lr': conservative_lr
                    }
                    adaptive_weight_log.append(log_entry)

                for k in weights.keys():
                    weight_history[k].append(weights[k])

        # -----Update learning rate based on epoch, not loss-----
        current_lr = lr_schedule[max([e for e in lr_schedule.keys() if e <= epoch])]
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # -----Switch to LBFGS for final fine-tuning to reduce oscillations-----
        if epoch == lbfgs_switch_epoch and epoch < start_epoch + n_epochs - 100:
            logger.info(f"Epoch {epoch}: Switching from Adam to LBFGS for final fine-tuning")
            lbfgs_params = curriculum_config.get('lbfgs_params', {}) if curriculum_config else {}
            lbfgs_optimizer = torch.optim.LBFGS(
                model.parameters(), 
                lr=lbfgs_params.get('lr', 1e-3),
                max_iter=lbfgs_params.get('max_iter', 20),
                history_size=lbfgs_params.get('history_size', 10),
                tolerance_grad=lbfgs_params.get('tolerance_grad', 1e-12),
                tolerance_change=lbfgs_params.get('tolerance_change', 1e-15)
            )
            optimizer = lbfgs_optimizer
            
            weight_update_freq = 50

        # -----Start of training step-----
        optimizer.zero_grad()

        # -----Forward pass with adaptive gradient strategy-----
        with torch.enable_grad():
            c_pred, time_points = model(c_initial=c_initial_epoch)
            if debug:
                debug_tensor_version("c_pred", c_pred)
                logger.debug("\n=== Prediction Diversity Check ===")
                for sample_idx in range(min(3, c_pred.shape[0])):
                    logger.debug(f"\nSample {sample_idx} predictions:")
                    time_indices = [0, c_pred.shape[1]//2, -1]
                    for t_idx in time_indices:
                        logger.debug(f"Time point {t_idx}: {c_pred[sample_idx, t_idx].detach().cpu().numpy()}")
        
            # -----Process predictions-----
            x_pred, env_pred = model.process_c_pred(
                c_pred=c_pred,
                env_dict=env_dict_epoch,
                time_points=time_points
            )

            # -----Get phase transformed concentrations-----
            c_transformed = model.get_phase_tranformed_c(
                x_pred=x_pred,
                time_points=time_points,
                env_pred=env_pred
            )

            # -----Get final predictions-----
            u_pred, c_pred_at_target = model.get_predictions(
                x_pred=x_pred,
                c_pred=c_pred,
                c_initial=c_initial_epoch,
                time_points=time_points,
                env_pred=env_pred
            )
        
            # -----Data loss-----
            if task == 'predict':
                loss_data = torch.tensor(0.0, device=device, dtype=torch.float64)
            elif run_mode == 'pinn+phase':
                if enable_data_shuffling:
                    loss_data = mse_loss(u_pred[:, 5:13], output_targets[valid_mask][shuffle_indices][:, 5:13])
                else:
                    loss_data = mse_loss(u_pred[:, 5:13], output_targets[valid_mask][:, 5:13])
            elif run_mode == 'pinn':
                loss_data = mse_loss(c_pred_at_target[:, :], c_final_epoch[:, 3:11])
            else:
                raise ValueError(f"Invalid run_mode: {run_mode}")
            
            # -----Initial condition loss-----
            if include_initial_loss:
                loss_initial_condition = mse_loss(c_pred[:, 0, :], c_initial_epoch[:, 3:11])
            else:
                loss_initial_condition = torch.tensor(0.0, device=device, dtype=torch.float64)
        
            # -----PDE loss-----
            r1_col, r2_col, r3_col, r4_col, r5_col, r6_col, r7_col = compute_pde_loss(
                model,
                c_initial=c_initial_epoch,
                time_points=time_points,
                env_pred=env_pred,
            )

            loss_pde_col = sum(torch.mean(r ** 2) for r in [r1_col, r2_col, r3_col, r4_col, r5_col, r6_col, r7_col])

            # -----Phase transformed loss-----
            loss_phase_transformed = mse_loss(c_transformed[:, 1:, :], c_pred[:, 1:, :])

            # -----Mass conservation penalty-----
            mass_conservation = compute_mass_conservation_penalty(
                c_pred=c_pred,
                c_initial=c_initial_epoch[:, 3:11]
            )

            # -----Safety checks for NaNs in individual losses-----
            if torch.isnan(loss_data) or torch.isinf(loss_data):
                loss_data = torch.tensor(1000.0, device=device, dtype=torch.float64)
                logger.warning("NaN detected in loss_data - using fallback value")

            if torch.isnan(loss_pde_col) or torch.isinf(loss_pde_col):
                loss_pde_col = torch.tensor(1000.0, device=device, dtype=torch.float64)
                logger.warning("NaN detected in loss_pde_col - using fallback value")

            if torch.isnan(loss_phase_transformed) or torch.isinf(loss_phase_transformed):
                loss_phase_transformed = torch.tensor(0.0, device=device, dtype=torch.float64)
                logger.warning("NaN detected in loss_phase_transformed - using fallback value")

            if torch.isnan(mass_conservation) or torch.isinf(mass_conservation):
                mass_conservation = torch.tensor(1000.0, device=device, dtype=torch.float64)
                logger.warning("NaN detected in mass_conservation - using fallback value")

            # -----Total loss-----
            if run_mode == 'pinn+phase' and epoch >= phase_intro_epoch:
                loss_components = [
                    weights['pde'] * loss_pde_col,
                    weights['mass'] * mass_conservation,
                    weights['phase'] * loss_phase_transformed,
                ]
                if task == 'train':
                    loss_components.append(weights['data'] * loss_data)
                if include_initial_loss:
                    loss_components.append(weights['initial'] * loss_initial_condition)
                loss = sum(loss_components)
            else:
                loss_components = [
                    weights['pde'] * loss_pde_col,
                    weights['mass'] * mass_conservation,
                ]
                if task == 'train':
                    loss_components.append(weights['data'] * loss_data)
                if include_initial_loss:
                    loss_components.append(weights['initial'] * loss_initial_condition)
                loss = sum(loss_components)
        
            if torch.isnan(loss).any():
                raise RuntimeError("NaN detected in loss")
            
            # -----Check for weight explosion after loss computation-----
            if 'weight_explosion_check_needed' in locals() and weight_explosion_check_needed:
                epochs_since_adaptive_start = epoch - curriculum_end_epoch
                if epochs_since_adaptive_start < 50:
                    adaptive_loss_threshold = 1e6  
                    threshold_phase = "very early"
                elif epochs_since_adaptive_start < 200:
                    adaptive_loss_threshold = 1e5
                    threshold_phase = "early"
                else:
                    adaptive_loss_threshold = 1e4
                    threshold_phase = "mature"
                
                is_exploded, explosion_msg = detect_weight_explosion(
                    current_weights=weights,
                    previous_weights=previous_weights,
                    loss_value=loss.item(),
                    threshold_factor=15.0,
                    loss_threshold=adaptive_loss_threshold
                )
                
                if adaptive_weight_log:
                    adaptive_weight_log[-1]['exploded'] = is_exploded
                    adaptive_weight_log[-1]['explosion_msg'] = explosion_msg
                
                if is_exploded:
                    logger.warning(f"WEIGHT EXPLOSION DETECTED at epoch {epoch}: {explosion_msg}")
                    logger.warning("Rolling back to stable weights and reducing learning rate")
                    
                    weights = stable_weights_backup.copy()
                    
                    emergency_lr = optimizer.param_groups[0]['lr'] * 0.5
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = emergency_lr
                    
                    logger.warning(f"Reduced learning rate to {emergency_lr:.2e} for stability")
                    
                    gradnorm_alpha = max(gradnorm_alpha * 0.8, 0.01)
                    gradnorm_lr = max(gradnorm_lr * 0.8, 0.0005)
                    
                    logger.warning(f"Reduced GradNorm parameters: α={gradnorm_alpha:.3f}, lr={gradnorm_lr:.4f}")
                
                weight_explosion_check_needed = False
            
            # -----Backward pass-----
            if task == 'train':
                phase_transition_epoch = curriculum_config.get('parameter_freezing', {}).get('phase_transition_epoch', 300) if curriculum_config else 300
                frozen_parameters = curriculum_config.get('parameter_freezing', {}).get('frozen_parameters', None) if curriculum_config else None
                toggle_parameter_freezing(model, epoch, phase_transition_epoch=phase_transition_epoch, frozen_parameters=frozen_parameters)

            loss.backward(retain_graph=True)

            if debug:
                logger.debug("\n=== Gradient Check ===")
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if len(param.grad.shape) > 1 and param.grad.shape[0] == c_initial_epoch.shape[0]:
                            for sample_idx in range(min(3, param.grad.shape[0])):
                                logger.debug(f"Sample {sample_idx} gradient for {name}:")
                                logger.debug(f"Mean: {param.grad[sample_idx].mean().item():.4e}")
                                logger.debug(f"Std: {param.grad[sample_idx].std().item():.4e}")
                        else:
                            logger.debug(f"Parameter {name} gradient statistics:")
                            logger.debug(f"Mean: {param.grad.mean().item():.4e}")
                            logger.debug(f"Std: {param.grad.std().item():.4e}")
            
            # -----Check for NaN in gradients-----
            has_nan_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    logger.warning(f"NaN gradient detected in {name}")
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                raise RuntimeError("NaN detected in gradients")
            
            # -----Optimizer step with different handling for LBFGS vs Adam-----
            if isinstance(optimizer, torch.optim.LBFGS):
                def closure():
                    optimizer.zero_grad()
                    
                    c_pred_closure, time_points_closure = model(c_initial=c_initial_epoch)
                    x_pred_closure, env_pred_closure = model.process_c_pred(
                        c_pred=c_pred_closure, env_dict=env_dict_epoch, time_points=time_points_closure)
                    c_transformed_closure = model.get_phase_tranformed_c(
                        x_pred=x_pred_closure, time_points=time_points_closure, env_pred=env_pred_closure)
                    u_pred_closure, c_pred_at_target_closure = model.get_predictions(
                        x_pred=x_pred_closure, c_pred=c_pred_closure, c_initial=c_initial_epoch,
                        time_points=time_points_closure, env_pred=env_pred_closure)
                    
                    if task == 'predict':
                        loss_data_closure = torch.tensor(0.0, device=device, dtype=torch.float64)
                    elif run_mode == 'pinn+phase':
                        if enable_data_shuffling:
                            loss_data_closure = mse_loss(u_pred_closure[:, 5:13], output_targets[valid_mask][shuffle_indices][:, 5:13])
                        else:
                            loss_data_closure = mse_loss(u_pred_closure[:, 5:13], output_targets[valid_mask][:, 5:13])
                    elif run_mode == 'pinn':
                        loss_data_closure = mse_loss(c_pred_at_target_closure[:, :], c_final_epoch[:, 3:11])
                    
                    if include_initial_loss:
                        loss_initial_condition_closure = mse_loss(c_pred_closure[:, 0, :], c_initial_epoch[:, 3:11])
                    else:
                        loss_initial_condition_closure = torch.tensor(0.0, device=device, dtype=torch.float64)
                    
                    r1_col_closure, r2_col_closure, r3_col_closure, r4_col_closure, r5_col_closure, r6_col_closure, r7_col_closure = compute_pde_loss(
                        model, c_initial=c_initial_epoch, time_points=time_points_closure, env_pred=env_pred_closure)
                    loss_pde_col_closure = sum(torch.mean(r ** 2) for r in [r1_col_closure, r2_col_closure, r3_col_closure, r4_col_closure, r5_col_closure, r6_col_closure, r7_col_closure])
                    
                    loss_phase_transformed_closure = mse_loss(c_transformed_closure[:, 1:, :], c_pred_closure[:, 1:, :])
                    mass_conservation_closure = compute_mass_conservation_penalty(c_pred=c_pred_closure, c_initial=c_initial_epoch[:, 3:11])
                    
                    if run_mode == 'pinn+phase' and epoch >= phase_intro_epoch:
                        loss_components_closure = [
                            weights['pde'] * loss_pde_col_closure,
                            weights['mass'] * mass_conservation_closure,
                            weights['phase'] * loss_phase_transformed_closure,
                        ]
                        if task == 'train':
                            loss_components_closure.append(weights['data'] * loss_data_closure)
                        if include_initial_loss:
                            loss_components_closure.append(weights['initial'] * loss_initial_condition_closure)
                        loss_closure = sum(loss_components_closure)
                    else:
                        loss_components_closure = [
                            weights['pde'] * loss_pde_col_closure,
                            weights['mass'] * mass_conservation_closure,
                        ]
                        if task == 'train':
                            loss_components_closure.append(weights['data'] * loss_data_closure)
                        if include_initial_loss:
                            loss_components_closure.append(weights['initial'] * loss_initial_condition_closure)
                        loss_closure = sum(loss_components_closure)
                    
                    loss_closure.backward(retain_graph=True)
                    return loss_closure
                
                optimizer.step(closure)
            else:
                optimizer.step()

        # -----Store history-----
        parameters_history.append(model.get_parameters())
        loss_history.append([loss.item(), loss_data.item(), loss_pde_col.item(),
                             loss_phase_transformed.item(), loss_initial_condition.item(),
                             mass_conservation.item(),
                             ])

        # -----Check if this is the best model so far-----
        is_best = loss.item() < best_loss
        if is_best:
            best_loss = loss.item()

        # -----Save checkpoint at specified frequency (guaranteed)-----
        should_save_periodic = ((epoch + 1) % checkpoint_freq == 0)
        should_save_final = (epoch == (start_epoch + n_epochs - 1))
        
        if should_save_periodic or should_save_final or is_best:
            save_checkpoint(
                epoch=epoch, 
                loss=loss.item(), 
                curr_weights=weights, 
                is_best=is_best,
                save_periodic=should_save_periodic,
                save_final=should_save_final
            )
            
            reasons = []
            if should_save_periodic:
                reasons.append(f"periodic (every {checkpoint_freq} epochs)")
            if should_save_final:
                reasons.append("final epoch")
            if is_best:
                reasons.append("best model")
            
            logger.info(f"Checkpoint trigger at epoch {epoch}: {', '.join(reasons)}")

        # -----Log phase loss introduction-----
        if epoch == phase_intro_epoch and run_mode == 'pinn+phase':
            logger.info(f"Epoch {epoch}: Introducing phase loss with initial weight {weights['phase']}")
        
        # -----Log training progress with GradNorm details-----
        if epoch % logging_freq == 0:
            weights_str = ", ".join([f"{k}: {v:.4e}" for k, v in weights.items() if k != 'initial' or include_initial_loss])
    
            log_components = [
                f'Epoch [{epoch}/{start_epoch + n_epochs}]',
                f'Total Loss: {loss.item():.4e}\n',
                f'PDE Loss: {loss_pde_col.item():.4e}',
                f'Mass Cons: {mass_conservation.item():.4e}',
            ]
            if task == 'train':
                log_components.insert(2, f'Data Loss: {loss_data.item():.4e}')
            if run_mode == 'pinn+phase':
                log_components.append(f'Phase Loss: {loss_phase_transformed.item():.4e}')
            
            if include_initial_loss:
                log_components.append(f'Initial Loss: {loss_initial_condition.item():.4e}')
            log_components.append(f'\nWeights: {weights_str}')
            log_components.append(f'LR: {optimizer.param_groups[0]["lr"]:.4e}')
            
            if epoch >= curriculum_end_epoch:
                log_components.append(f'Method: GradNorm (α={gradnorm_alpha}, lr={gradnorm_lr})')
                
                target_rates_str = ", ".join([f"{k}: {v:.2f}" for k, v in target_rates.items() if k in weights])
                log_components.append(f'Target Rates: {target_rates_str}')
            
            logger.info(', '.join(log_components))
            # -----Log gradient norms for different parameter groups-----
            reaction_grad_norm = 0
            species_grad_norm = 0
            network_grad_norm = 0
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if name.startswith('x_A_'):
                        reaction_grad_norm += param.grad.norm().item() ** 2
                    elif name.startswith('z_Delta_H_') or name.startswith('y_E_'):
                        species_grad_norm += param.grad.norm().item() ** 2
                    else:
                        network_grad_norm += param.grad.norm().item() ** 2
            
            reaction_grad_norm = reaction_grad_norm ** 0.5
            species_grad_norm = species_grad_norm ** 0.5
            network_grad_norm = network_grad_norm ** 0.5
            
            logger.info(f'Gradient Norms - Reaction: {reaction_grad_norm:.4e}, '
                        f'Species: {species_grad_norm:.4e}, Network: {network_grad_norm:.4e}')

            # -----Log concentration metrics-----
            with torch.no_grad():
                c_mean = torch.mean(torch.abs(c_pred[:, 1:, :])).item()
                logger.info(f"C_pred mean abs: {c_mean:.4e}")
                
                for t_idx in [1, len(time_points)//2, len(time_points)-2]:
                    species_values = []
                    for s_idx in range(min(3, c_pred.shape[2])):  # First 3 species
                        species_values.append(f"C{s_idx}={c_pred[0, t_idx, s_idx].item():.4e}")
                    
                    time_val = time_points[t_idx].item()
                    logger.info(f"t={time_val:.4e}: " + ", ".join(species_values))
        
        # -----Clean up CUDA cache periodically-----
        if epoch % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -----Plot weight history and GradNorm metrics after training-----
    try:            
        valid_weights = {}
        for k, v in weight_history.items():
            if len(v) > 0:
                valid_weights[k] = v
        
        if valid_weights and any(len(v) > 0 for v in valid_weights.values()):  # Make sure we have at least one valid series with data
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            ax1 = axes[0, 0]
            weight_lengths = [len(v) for v in valid_weights.values()]
            if all(length > 0 for length in weight_lengths):
                has_valid_data = False
                for k, v in valid_weights.items():
                    if len(v) > 0:
                        weight_epochs = list(range(len(v)))
                        v_array = np.array(v)
                        if np.all(np.isfinite(v_array)) and np.all(v_array > 0):
                            try:
                                ax1.semilogy(weight_epochs, v, label=f'{k} weight', linewidth=2)
                                has_valid_data = True
                            except Exception as e:
                                logger.warning(f"Failed to plot weight {k}: {e}")
                
                if has_valid_data:
                    ax1.set_xlabel('Update Steps')
                    ax1.set_ylabel('Weight Value (log scale)')
                    ax1.set_title('GradNorm Weight Evolution')
                    ax1.grid(True, which='both', linestyle='--', alpha=0.6)
                    ax1.legend()
                else:
                    ax1.text(0.5, 0.5, 'No valid weight data for plotting', 
                            ha='center', va='center', transform=ax1.transAxes)
            else:
                ax1.text(0.5, 0.5, 'No weight evolution data available', 
                        ha='center', va='center', transform=ax1.transAxes)
            
            ax2 = axes[0, 1]
            valid_grads = {}
            for k, v in grad_history.items():
                if len(v) > 0:
                    valid_grads[k] = v
            
            if valid_grads:
                has_valid_grad_data = False
                for k, v in valid_grads.items():
                    if len(v) > 0:
                        grad_epochs = list(range(len(v)))
                        v_array = np.array(v)
                        if np.all(np.isfinite(v_array)) and np.all(v_array > 0):
                            try:
                                ax2.semilogy(grad_epochs, v, label=f'{k} gradient', linewidth=2)
                                has_valid_grad_data = True
                            except Exception as e:
                                logger.warning(f"Failed to plot gradient {k}: {e}")
                
                if has_valid_grad_data:
                    ax2.set_xlabel('Update Steps')
                    ax2.set_ylabel('Gradient Magnitude (log scale)')
                    ax2.set_title('Gradient Magnitude Evolution')
                    ax2.grid(True, which='both', linestyle='--', alpha=0.6)
                    ax2.legend()
                else:
                    ax2.text(0.5, 0.5, 'No valid gradient data for plotting', 
                            ha='center', va='center', transform=ax2.transAxes)
            else:
                ax2.text(0.5, 0.5, 'No gradient data available', 
                        ha='center', va='center', transform=ax2.transAxes)
            
            ax3 = axes[1, 0]
            if adaptive_weight_log:
                target_rates_plot = {}
                for entry in adaptive_weight_log:
                    if 'target_rates' in entry:
                        for task, rate in entry['target_rates'].items():
                            if task not in target_rates_plot:
                                target_rates_plot[task] = []
                            target_rates_plot[task].append(rate)
                
                if target_rates_plot and any(len(rates) > 0 for rates in target_rates_plot.values()):
                    for task, rates in target_rates_plot.items():
                        if len(rates) > 0:
                            ax3.plot(rates, label=f'{task} target', linewidth=2)
                    
                    ax3.set_xlabel('Update Steps')
                    ax3.set_ylabel('Target Training Rate')
                    ax3.set_title('GradNorm Target Rates')
                    ax3.grid(True, linestyle='--', alpha=0.6)
                    ax3.legend()
                else:
                    ax3.text(0.5, 0.5, 'No target rate data available', 
                            ha='center', va='center', transform=ax3.transAxes)
            
            ax4 = axes[1, 1]
            if adaptive_weight_log and len(adaptive_weight_log) > 1:
                weight_changes = {}
                for i, entry in enumerate(adaptive_weight_log[1:], 1):
                    prev_entry = adaptive_weight_log[i-1]
                    if 'weights' in entry and 'weights' in prev_entry:
                        for task in entry['weights']:
                            if task in prev_entry['weights']:
                                if task not in weight_changes:
                                    weight_changes[task] = []
                                change = abs(entry['weights'][task] - prev_entry['weights'][task])
                                weight_changes[task].append(change)
                
                if weight_changes and any(len(changes) > 0 for changes in weight_changes.values()):
                    for task, changes in weight_changes.items():
                        if len(changes) > 0:
                            ax4.semilogy(changes, label=f'{task} change', linewidth=2)
                    
                    ax4.set_xlabel('Update Steps')
                    ax4.set_ylabel('Weight Change Magnitude (log scale)')
                    ax4.set_title('Weight Adaptation Dynamics')
                    ax4.grid(True, which='both', linestyle='--', alpha=0.6)
                    ax4.legend()
                else:
                    ax4.text(0.5, 0.5, 'No weight change data available', 
                            ha='center', va='center', transform=ax4.transAxes)
            else:
                ax4.text(0.5, 0.5, 'Insufficient data for weight changes', 
                        ha='center', va='center', transform=ax4.transAxes)
            
            plt.tight_layout()
            plot_path = os.path.join(checkpoint_dir, f'gradnorm_analysis.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"GradNorm analysis plot saved to: {plot_path}")
            plt.close()
            
            plt.figure(figsize=(12, 8))
            has_simple_plot_data = False
            
            for k, v in valid_weights.items():
                if len(v) > 0:
                    epochs = list(range(len(v)))
                    v_array = np.array(v)
                    if np.all(np.isfinite(v_array)) and np.all(v_array > 0):
                        try:
                            plt.semilogy(epochs, v, label=f'{k} weight', linewidth=2)
                            has_simple_plot_data = True
                        except Exception as e:
                            logger.warning(f"Failed to plot weight {k} in simple plot: {e}")
            
            if has_simple_plot_data:
                plt.xlabel('Update Steps')
                plt.ylabel('Weight Value (log scale)')
                plt.title('GradNorm Adaptive Weight Evolution')
                plt.grid(True, which='both', linestyle='--', alpha=0.6)
                plt.legend()
                
                plot_path = os.path.join(checkpoint_dir, f'adaptive_weights.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                logger.info(f"Adaptive weight history plot saved to: {plot_path}")
                plt.close()
            else:
                plt.close()  # Close the empty figure
                logger.warning("No valid data for simple weight plot")
        else:
            logger.warning("No weight history data available to plot")
            
    except Exception as e:
        logger.error(f"Failed to plot GradNorm analysis: {str(e)}")
        logger.error(f"Weight history keys: {list(weight_history.keys())}")
        for k, v in weight_history.items():
            logger.error(f"Weight '{k}' has {len(v)} entries")
        for k, v in grad_history.items():
            logger.error(f"Gradient '{k}' has {len(v)} entries")

    return parameters_history, loss_history

def gradnorm_update_weights(weights, grad_magnitudes, loss_terms, target_rates, alpha=0.12, learning_rate=0.025, 
                           loss_history=None, epoch=None):
    """
    Advanced GradNorm-based adaptive weighting strategy
    
    Args:
        weights: Current loss weights
        grad_magnitudes: Gradient magnitudes for each loss term
        loss_terms: Current loss values
        target_rates: Target relative training rates for each task
        alpha: GradNorm hyperparameter (restoring force strength)
        learning_rate: Weight update learning rate
        loss_history: Historical loss values for computing training rates
        epoch: Current epoch number
    
    Returns:
        Updated weights
    """
    r_i = {}
    min_epoch_req = 3 if any(len(v) > 10 for v in loss_history.values()) else 10
    min_history_req = 3 if any(len(v) > 10 for v in loss_history.values()) else 5
    
    if loss_history and epoch and epoch > min_epoch_req:
        for task in weights.keys():
            if task in loss_history and len(loss_history[task]) >= min_history_req:
                current_loss = loss_history[task][-1]
                window_size = min(10, max(3, len(loss_history[task]) // 3))
                past_loss = np.mean(loss_history[task][-window_size-3:-3]) if len(loss_history[task]) >= window_size + 3 else loss_history[task][0]
                
                if past_loss > 1e-10 and current_loss > 1e-10:
                    rate = (past_loss - current_loss) / (past_loss + 1e-8)
                    r_i[task] = max(0.01, min(2.0, rate))
                else:
                    r_i[task] = target_rates.get(task, 1.0) / 5.0
            else:
                r_i[task] = target_rates.get(task, 1.0) / 5.0
    else:
        for task in weights.keys():
            r_i[task] = target_rates.get(task, 1.0) / 5.0
    
    if r_i:
        avg_rate = np.mean(list(r_i.values()))
        avg_rate = max(avg_rate, 1e-6)
    else:
        return weights
    
    grad_norm_targets = {}
    grad_values = list(grad_magnitudes.values())
    if not grad_values:
        return weights
    
    median_grad = np.median(grad_values) if len(grad_values) > 1 else grad_values[0]
    mean_grad = np.mean(grad_values)
    avg_grad = 0.7 * median_grad + 0.3 * mean_grad
    avg_grad = max(avg_grad, 1e-10)
    
    for task in weights.keys():
        if task in grad_magnitudes and task in r_i:
            target_rel_rate = target_rates.get(task, 1.0)
            
            relative_rate = r_i[task] / avg_rate
            relative_rate = np.clip(relative_rate, 0.2, 5.0)
            
            alpha_damped = min(alpha, 0.5)
            grad_norm_targets[task] = avg_grad * (relative_rate ** alpha_damped) * target_rel_rate
    
    new_weights = {}
    max_weight_change = 0.3
    
    for task in weights.keys():
        if task in grad_magnitudes and task in grad_norm_targets:
            current_grad = max(grad_magnitudes[task], 1e-10)
            target_grad = max(grad_norm_targets[task], 1e-10)
            
            grad_ratio = current_grad / target_grad
            grad_ratio = np.clip(grad_ratio, 0.1, 10.0)
            
            if grad_ratio > 1.0:
                adaptive_lr = learning_rate / (1.0 + 0.5 * (grad_ratio - 1.0))
            else:
                adaptive_lr = learning_rate * (0.5 + 0.5 * grad_ratio)
            
            weight_update = adaptive_lr * np.tanh(1 - grad_ratio)
            weight_update = np.clip(weight_update, -max_weight_change, max_weight_change)
            
            log_weight = np.log(max(weights[task], 1e-10))
            new_log_weight = log_weight + weight_update
            new_weights[task] = np.exp(new_log_weight)
            
            change_ratio = new_weights[task] / weights[task]
            if change_ratio > 1.5:
                new_weights[task] = weights[task] * 1.5
            elif change_ratio < 0.67:
                new_weights[task] = weights[task] * 0.67
                
        else:
            new_weights[task] = weights[task]
    
    total_old = sum(weights.values())
    total_new = sum(new_weights.values())
    if total_new > 0 and total_old > 0:
        scale_factor = total_old / total_new
        scale_factor = np.clip(scale_factor, 0.8, 1.2)
        
        for task in new_weights:
            new_weights[task] = new_weights[task] * scale_factor
    
    return new_weights