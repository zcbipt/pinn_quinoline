# ------------------------------------------------------------------------------------------
# Physics-Informed Neural Network (PINN) Model Architecture for Quinoline Synthesis
# 
# This module implements the core PINN model architecture that combines neural networks
# with physical laws for quinoline synthesis prediction. Features include thermodynamic
# parameter learning, species concentration modeling, and physics-constrained predictions.
# 
# Contributions welcome! Please see CONTRIBUTING.md for guidelines.
# ------------------------------------------------------------------------------------------

import os
import sys
from loguru import logger

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(ROOT_DIR)
CONFIG_DIR = os.path.join(ROOT_DIR, "config")
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "model")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

from .plot import plot_trajectory

import torch
import torch.nn as nn
from torch.serialization import add_safe_globals
torch.set_default_dtype(torch.float64)
add_safe_globals([nn.Sequential, nn.Linear, nn.ReLU, nn.Tanh, nn.Softplus])


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, dim)
        )
        self.activation = nn.Tanh()
        
        for m in self.block.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=5/3)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.activation(x + self.block(x))


class TimeEncoder(nn.Module):
    def __init__(self, encoder_dim=16):
        super().__init__()
        self.time_net = nn.Sequential(
            nn.Linear(1, encoder_dim),
            nn.Tanh(),
            nn.Linear(encoder_dim, encoder_dim),
        )
        
        for m in self.time_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=5/3)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, t):
        return self.time_net(t)


class SampleEncoder(nn.Module):
    def __init__(self, encoder_dim=32):
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(13, 64),
            nn.Tanh(),
            nn.Linear(64, encoder_dim),
        )
        
        self.skip_connection = nn.Linear(13, encoder_dim)
        
        for m in self.feature_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=5/3)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        nn.init.xavier_uniform_(self.skip_connection.weight, gain=0.3)
        nn.init.zeros_(self.skip_connection.bias)
    
    def forward(self, x):
        real_features = x[:, :13]
        feature_encoding = self.feature_net(real_features)
        
        skip_output = self.skip_connection(real_features)
        
        batch_size = x.shape[0]
        unique_encodings = torch.zeros_like(feature_encoding)
        
        for i in range(batch_size):
            sample_features = real_features[i]
            
            weights = torch.tensor(
                [1.0, -1.5, 1.2, -0.9, 0.8, -1.2, 1.0, -0.7, 1.3, -0.5, 0.9, -1.1, 0.6], 
                device=x.device
            )
            
            feature_hash_1 = torch.sum(sample_features * weights)
            
            feature_hash_2 = torch.sum(sample_features * torch.sin(torch.linspace(0.0, 6.28, 13, device=x.device)))
            
            for j in range(feature_encoding.shape[1]):
                frequency = 3.0 + (j % 7) * 0.8
                phase_shift = (j % 5) * 0.5
                unique_encodings[i, j] = 1.0 * torch.sin(
                    frequency * torch.tensor(j) * feature_hash_1 / 5.0 + feature_hash_2 + phase_shift
                )
        
        alpha = 0.6
        beta = 0.2
        gamma = 0.2
        
        total = alpha + beta + gamma
        alpha = alpha / total
        beta = beta / total
        gamma = gamma / total
        
        mixed_encoding = alpha * feature_encoding + beta * unique_encodings + gamma * skip_output
        
        return mixed_encoding


class FeatureFusion(nn.Module):
    def __init__(self, time_encoder_dim=32, sample_encoder_dim=16):
        super().__init__()
        
        self.time_proj = nn.Linear(time_encoder_dim, time_encoder_dim)
        
        self.sample_proj = nn.Linear(sample_encoder_dim, sample_encoder_dim)
        
        self.time_norm = nn.LayerNorm(time_encoder_dim)
        self.sample_norm = nn.LayerNorm(sample_encoder_dim)
        
        self.mixer = nn.Sequential(
            nn.Linear(time_encoder_dim + sample_encoder_dim, time_encoder_dim + sample_encoder_dim),
            nn.GELU(),
            nn.Linear(time_encoder_dim + sample_encoder_dim, time_encoder_dim + sample_encoder_dim),
        )
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m is self.sample_proj:
                    nn.init.xavier_uniform_(m.weight, gain=1.5)
                elif m in self.mixer:
                    nn.init.xavier_uniform_(m.weight, gain=1.41)
                else:
                    nn.init.xavier_uniform_(m.weight, gain=1.0)
                    
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, time_features, sample_features):
        batch_size = time_features.shape[0]
        
        time_proj = self.time_norm(self.time_proj(time_features))
        sample_proj = self.sample_norm(self.sample_proj(sample_features))
        
        time_orig = time_features
        sample_orig = sample_features
        
        combined = torch.cat([sample_proj, time_proj], dim=1)
        
        mixed = self.mixer(combined)
        
        orig_combined = torch.cat([sample_orig, time_orig], dim=1)
        output = mixed + 0.5 * orig_combined
        
        return output


class PINN(nn.Module):
    def __init__(self, num_inputs=None, num_outputs=None, delta_t=None,
                 device=None, time_batch_size=None, env_smoothing_factor=None, pinn_inputs=None, training_flag=None):
        """
        Args:
            num_inputs: number of inputs to the PINN
            num_outputs: number of outputs from the PINN
            delta_t: time step for the PINN
            device: device to run the PINN on
            time_batch_size: number of time points to process in each batch
            env_smoothing_factor: factor to smooth the environment
            pinn_inputs: inputs to the PINN (['time', 'time+initials'])
            training_flag: flag to indicate if the PINN is in training mode
        """
        super().__init__()

        assert pinn_inputs in ['time', 'time+initials'], f"Invalid PINN inputs: {pinn_inputs}"

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        self.config = {
            'num_inputs': num_inputs,
            'num_outputs': num_outputs,
            'delta_t': delta_t,
            'time_batch_size': time_batch_size,
            'env_smoothing_factor': env_smoothing_factor,
            'pinn_inputs': pinn_inputs,
            'training_flag': training_flag
        }

        self.delta_t = delta_t
        self.epsilon = 1e-10
        self.time_batch_size = time_batch_size
        self.env_smoothing_factor = env_smoothing_factor
        self.pinn_inputs = pinn_inputs
        self.training_flag = training_flag

        self.time_encoder_dim = 16
        self.sample_encoder_dim = 16

        if pinn_inputs == 'time':
            self.num_inputs_pinn = 1
        else:
            self.num_inputs_pinn = self.time_encoder_dim + self.sample_encoder_dim
        
        self.sample_encoder = SampleEncoder(encoder_dim=self.sample_encoder_dim)
        self.time_encoder = TimeEncoder(encoder_dim=self.time_encoder_dim)
        self.feature_fusion = FeatureFusion(time_encoder_dim=self.time_encoder_dim, sample_encoder_dim=self.sample_encoder_dim)

        self.net = nn.Sequential(
            nn.Linear(self.num_inputs_pinn, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, num_outputs),
            nn.Softplus(),
        )
        
        for i, module in enumerate(self.net):
            if isinstance(module, nn.Linear):
                if i < len(self.net) - 2 and isinstance(self.net[i+2], nn.Tanh):
                    nn.init.xavier_uniform_(module.weight, gain=5/3)
                elif i == len(self.net) - 2:
                    nn.init.xavier_uniform_(module.weight, gain=1.2)
                else:
                    nn.init.xavier_uniform_(module.weight, gain=1.0)
                    
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.sample_encoder.to(self.device)
        self.time_encoder.to(self.device)
        self.feature_fusion.to(self.device)
        self.net.to(self.device)

        reaction_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        species_indices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']

        for i in reaction_indices:
            if i <= 3:
                mean = 0.3 + 0.1 * i
                std = 0.05
            elif i <= 6:
                mean = 0.1 + 0.05 * (i-3)
                std = 0.03
            else:
                mean = 0.05 + 0.02 * (i-6)
                std = 0.01
                
            init_value = mean + std * torch.randn(1)
            setattr(self, f'x_A_{i}', nn.Parameter(init_value))

        for i in reaction_indices:
            if i % 3 == 0:
                mean = -1.5
                std = 1.0
            elif i % 3 == 1:
                mean = 1.5
                std = 1.0
            else:
                mean = 0.0
                std = 1.5
            
            init_value = mean + std * torch.randn(1)
            init_value = torch.clamp(init_value, -4.0, 4.0)
            setattr(self, f'y_E_{i}', nn.Parameter(init_value))

        for i, s in enumerate(species_indices):
            mean_a = 0.01 * (i+1)
            std_a = 0.005
            init_value_a = mean_a + std_a * torch.randn(1)
            setattr(self, f'x_A_{s}', nn.Parameter(init_value_a))
            
            if i % 3 == 0:
                mean_h = -1.0
                std_h = 0.8
            elif i % 3 == 1:
                mean_h = 1.0
                std_h = 0.8
            else:
                mean_h = 0.0
                std_h = 1.2
            
            init_value_h = mean_h + std_h * torch.randn(1)
            init_value_h = torch.clamp(init_value_h, -4.0, 4.0)
            setattr(self, f'z_Delta_H_{s}', nn.Parameter(init_value_h))
        
        self.to(self.device)

    def to(self, device):
        """Override to() method to properly move all components including phase model to device"""
        super().to(device)
        
        self.device = device
        
            
        return self

    def preprocess_initial_inputs(self, x):
        """
        Preprocess raw inputs to generate time, pH, and initial concentrations
        
        Args:
            x: Raw input features [batch_size, n_features]
        
        Returns:
            c_initial: tensor with processed inputs for PINN
        """
        x = x.to(self.device)
        
        t = x[:, 0].clone()
        T = x[:, 1]
        P = x[:, 2]
        FEED = x[:, 3]
        FH2 = x[:, 4]
        
        c_phase = x[:, 5:13]

        pH = x[:, 13]
        FL = torch.ones(x.shape[0], dtype=torch.float64, device=self.device)
        VV = torch.ones(x.shape[0], dtype=torch.float64, device=self.device)
        CS = torch.ones(x.shape[0], dtype=torch.float64, device=self.device)

        CA = c_phase[:, 0]
        CB = c_phase[:, 1]
        CC = c_phase[:, 2]
        CD = c_phase[:, 3]
        CE = c_phase[:, 4]
        CF = c_phase[:, 5]
        CG = c_phase[:, 6]
        CI = c_phase[:, 7]

        YA = torch.ones(x.shape[0], dtype=torch.float64, device=self.device)
        YB = torch.ones(x.shape[0], dtype=torch.float64, device=self.device)
        YC = torch.ones(x.shape[0], dtype=torch.float64, device=self.device)
        YD = torch.ones(x.shape[0], dtype=torch.float64, device=self.device)
        YE = torch.ones(x.shape[0], dtype=torch.float64, device=self.device)
        YF = torch.ones(x.shape[0], dtype=torch.float64, device=self.device)
        YI = torch.ones(x.shape[0], dtype=torch.float64, device=self.device)

        c_initial = torch.stack([t, T, pH, CA, CB, CC, CD, CE, CF, CG, CI, P, FEED, FH2], dim=1)
        c_initial.requires_grad_(True)

        env_dict = {
            't': t,
            'T': T,
            'P': P,
            'FEED': FEED,
            'FH2': FH2,
            'pH': pH,
            'FL': FL,
            'VV': VV,
            'CS': CS,
            'YA': YA,
            'YB': YB,
            'YC': YC,
            'YD': YD,
            'YE': YE,
            'YF': YF,
            'YI': YI
        }

        return c_initial, env_dict
    

    def forward(self, c_initial=None, debug=False):
        """
        Forward pass that predicts C* concentrations at regular time intervals
        
        Args:
            c_initial: Input tensor [batch_size, num(t, T, pH, CA, CB, CC, CD, CE, CF, CG, CI, P, FEED, FH2)]
            debug: Whether to print debug information
            
        Returns:
            c_pred: Predicted C* concentrations [batch_size, n_time_points, 8]
            time_points: Tensor of time points [n_time_points]
        """
        batch_size = c_initial.shape[0]

        max_target_time = c_initial[:, 0].max()
        normalized_target_time = c_initial[:, 0] / (max_target_time + self.epsilon)
        temporal_context = normalized_target_time.unsqueeze(1)

        sample_features = torch.cat([
            c_initial[:, 1:2],   # T
            c_initial[:, 2:3],   # pH
            c_initial[:, 3:11],  # initial_conc
            c_initial[:, 11:14], # P, FEED, FH2
            temporal_context,    # Normalized target time
        ], dim=1)

        # -----Encode sample features once-----
        sample_encoding = self.sample_encoder(sample_features)  # [batch_size, 24]
        # Check if sample encodings are actually different
        if c_initial.shape[0] > 1:
            while True:
                idx1, idx2 = torch.randint(0, c_initial.shape[0], (2,))
                if idx1 != idx2:
                    break
            diff = torch.mean(torch.abs(sample_encoding[idx1] - sample_encoding[idx2]))
            if diff.item() < 1e-6:
                logger.warning("WARNING: Sample encodings are nearly identical!")
        
       # -----Get target times for each sample-----
        target_times = c_initial[:, 0]
        
        # -----Calculate number of time steps needed for each sample-----
        n_steps = torch.ceil(target_times / self.delta_t).long()
        max_steps = n_steps.max().item()
        
        # -----Process all time points (including time 0)-----
        time_batches_results = []
        
        # -----Process time points in batches to reduce memory usage-----
        time_batch_size = self.time_batch_size
        
        for t_start in range(0, max_steps + 1, time_batch_size):
            t_end = min(t_start + time_batch_size, max_steps + 1)
            
            for t_idx in range(t_start, t_end):
                t_relative = torch.ones(batch_size, dtype=torch.float64, device=self.device) * (t_idx * self.delta_t)
                t_batch_tensor = t_relative.unsqueeze(1)
                t_batch_tensor.requires_grad_(True)

                # -----Encode time-----
                time_encoding = self.time_encoder(t_batch_tensor)  # [batch_size, time_encoder_dim]
                
                if self.pinn_inputs == 'time':
                    species_outputs = self.net(t_batch_tensor)
                elif self.pinn_inputs == 'time+initials':
                    combined_features = self.feature_fusion(time_encoding, sample_encoding)
                    species_outputs = self.net(combined_features)

                time_batches_results.append(species_outputs)
        
        c_pred_final = torch.stack(time_batches_results, dim=1)

        time_points = torch.linspace(0, max_steps * self.delta_t, max_steps + 1, 
                                    dtype=torch.float64, device=self.device)
        
        return c_pred_final, time_points
    
    def process_c_pred(self, c_pred=None, env_dict=None, time_points=None):
        """
        Process predicted concentrations and environment dictionary to get x_pred and env_pred
        
        Args:
            c_pred: Predicted concentrations [batch_size, n_time_points, 8]
            env_dict: Environment dictionary
            time_points: Tensor of time points [n_time_points]
        Returns:
            x_pred: Predicted x values [batch_size, n_time_points, 8]
            env_pred: Predicted environment values [batch_size, n_time_points, n_env_vars]
        """
        device = c_pred.device

        # -----Convert all inputs to the same device if needed-----
        env_dict = {k: v.to(device) for k, v in env_dict.items()}
        time_points = time_points.to(device)

        batch_size = c_pred.shape[0]
        n_time_points = len(time_points)

        # -----Extract initial environment parameters-----
        T_initial = env_dict['T']
        P_initial = env_dict['P']
        FEED_initial = env_dict['FEED']
        FH2_initial = env_dict['FH2']
        pH_initial = env_dict['pH']
        FL_initial = env_dict['FL']
        VV_initial = env_dict['VV']
        CS_initial = env_dict['CS']
        YA_initial = env_dict['YA']
        YB_initial = env_dict['YB']
        YC_initial = env_dict['YC']
        YD_initial = env_dict['YD']
        YE_initial = env_dict['YE']
        YF_initial = env_dict['YF']
        YI_initial = env_dict['YI']

        # -----Initialize storage for results-----
        env_pred_list = []

        # -----Initial environment state - create a copy to avoid in-place modification-----
        env_initial = torch.stack([T_initial, P_initial, FEED_initial, FH2_initial,
                                   pH_initial, FL_initial, VV_initial, CS_initial,
                                   YA_initial, YB_initial, YC_initial, YD_initial,
                                   YE_initial, YF_initial, YI_initial], dim=1) 

        env_pred_list.append(env_initial)

        # -----For each time point (except t=0), predict concentrations-----
        for t_idx in range(1, n_time_points):
            with torch.enable_grad():  # Ensure we track gradients properly
                # -----Current time-----
                t = time_points[t_idx]

                # -----Get current environment parameters from previous step-----
                current_env = env_pred_list[-1].clone()
            
                # -----Extract values from current environment-----
                current_T = current_env[:, 0]
                current_P = current_env[:, 1]
                current_FEED = current_env[:, 2]
                current_FH2 = current_env[:, 3]
                current_pH = pH_initial
                current_FL = current_env[:, 5]
                current_VV = current_env[:, 6]
                current_CS = current_env[:, 7]
                current_YA = current_env[:, 8]
                current_YB = current_env[:, 9]
                current_YC = current_env[:, 10]
                current_YD = current_env[:, 11]
                current_YE = current_env[:, 12]
                current_YF = current_env[:, 13]
                current_YI = current_env[:, 14]

                # -----Predict environment parameters-----
                t_batch = torch.ones(batch_size, 1, dtype=torch.float64, device=self.device) * t
                t_batch.requires_grad_(True)

                pH_current = current_pH
                FL_current = current_FL
                VV_current = current_VV
                CS_current = current_CS

                YA_current = current_YA
                YB_current = current_YB
                YC_current = current_YC
                YD_current = current_YD
                YE_current = current_YE
                YF_current = current_YF
                YI_current = current_YI

                # -----Create a fresh tensor for the new environment state-----
                env_new = torch.stack([
                    current_T, current_P, current_FEED, current_FH2, 
                    pH_current, FL_current, VV_current, CS_current,
                    YA_current, YB_current, YC_current, YD_current, 
                    YE_current, YF_current, YI_current
                ], dim=1)
                
                env_pred_list.append(env_new)

        # -----Create the final tensors by stacking the lists-----
        env_pred = torch.stack(env_pred_list, dim=1)

        return env_pred
    

    def get_predictions(self, c_pred=None, c_initial=None, time_points=None):
        """
        Get predictions for both x_pred and c_pred based on target time in c_initial
        
        Args:
            c_pred: Predicted C* concentrations [batch_size, n_time_points, 8]
            c_initial: Initial concentrations [batch_size, n_features]
            time_points: Tensor of time points [n_time_points]
        Returns:
            c_pred_at_target: Predicted C* values at target times [batch_size, 8]
        """
        batch_size = c_initial.shape[0]

        # -----Extract target times from c_initial-----
        target_times = c_initial[:, 0]
        
        # -----Initialize output tensors-----
        c_pred_at_target = torch.zeros((batch_size, c_pred.shape[-1]), dtype=torch.float64, device=self.device)
        
        # -----Process each sample independently-----
        for i in range(batch_size):
            t_target = target_times[i]
            
            time_diffs = torch.abs(time_points - t_target)
            closest_idx = torch.argmin(time_diffs).item()

            if time_points[closest_idx] > t_target and closest_idx > 0:
                closest_idx -= 1

            c_pred_at_target[i] = c_pred[i, closest_idx]
        
        return c_pred_at_target


    def get_parameters(self):
        return self.x_A_1.item(), self.x_A_2.item(), self.x_A_3.item(), self.x_A_4.item(), self.x_A_5.item(), self.x_A_6.item(), self.x_A_7.item(), self.x_A_8.item(), self.x_A_9.item(), self.x_A_10.item(),\
               self.x_A_A.item(), self.x_A_B.item(), self.x_A_C.item(), self.x_A_D.item(), self.x_A_E.item(), self.x_A_F.item(), self.x_A_G.item(), self.x_A_H.item(), self.x_A_I.item(),\
               self.y_E_1.item(), self.y_E_2.item(), self.y_E_3.item(), self.y_E_4.item(), self.y_E_5.item(), self.y_E_6.item(), self.y_E_7.item(), self.y_E_8.item(), self.y_E_9.item(), self.y_E_10.item(),\
               self.z_Delta_H_A.item(), self.z_Delta_H_B.item(), self.z_Delta_H_C.item(), self.z_Delta_H_D.item(), self.z_Delta_H_E.item(), self.z_Delta_H_F.item(), self.z_Delta_H_G.item(), self.z_Delta_H_H.item(), self.z_Delta_H_I.item()


    def save_model(self, save_path):
        """
        Save PINN model with all necessary information
        
        Args:
            save_path: path to save model file
        """
        try:
            save_dir = os.path.dirname(save_path)
            
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                logger.info(f"Created directory: {save_dir}")
            
            model_save = {
                'state_dict': self.state_dict(),
                'config': self.config
            }
            torch.save(model_save, save_path)
            logger.info(f"Model saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    @classmethod
    def load_model(cls, model_path):
        """
        Load saved PINN model
        
        Args:
            model_path: path to saved model file
        Returns:
            loaded PINN model
        """
        # -----Determine the appropriate device-----
        map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # -----Load the checkpoint with device mapping-----
        checkpoint = torch.load(model_path, map_location=map_location)

        device = map_location if isinstance(map_location, torch.device) else torch.device(map_location)
        
        # -----Initialize model with saved config-----
        if 'pinn_inputs' in checkpoint['config'].keys():
            model = cls(
                num_inputs=checkpoint['config']['num_inputs'],
                num_outputs=checkpoint['config']['num_outputs'],
                delta_t=checkpoint['config']['delta_t'],
                time_batch_size=checkpoint['config']['time_batch_size'],
                env_smoothing_factor=checkpoint['config']['env_smoothing_factor'],
                pinn_inputs=checkpoint['config']['pinn_inputs'],
                training_flag=checkpoint['config']['training_flag'],
                device=device
            )
        else:
            model = cls(
                num_inputs=checkpoint['config']['num_inputs'],
                num_outputs=checkpoint['config']['num_outputs'],
                delta_t=checkpoint['config']['delta_t'],
                time_batch_size=checkpoint['config']['time_batch_size'],
                env_smoothing_factor=checkpoint['config']['env_smoothing_factor'],
                pinn_inputs='time',
                training_flag=False,
                device=device
            )
        
        # -----Load state dict-----
        model.load_state_dict(checkpoint['state_dict'])
        
        # -----Set to evaluation mode-----
        model.eval()
        
        logger.info(f"Model loaded from {model_path} to device: {device}")
        return model
    
    def inference(self, input_features=None, output_subfolder=None, c_scaling_factor=None, debug=False):
        """
        Run model inference and create plots for concentration changes over time
        
        Args:
            input_features: Tensor with input features [batch_size, n_features]
                        First column contains target time for each sample
            output_subfolder: Name of subfolder to save plots (under OUTPUT_DIR)
            
        Returns:
            predictions: Tensor of predicted concentrations at target times
            all_predictions: Dictionary mapping sample index to tensor of predictions at all time points
        """
        self.training_flag = False
        self.eval()

        # -----Move input to device-----
        input_features = input_features.to(self.device)

        if output_subfolder is not None:
            logger.info(f"output_subfolder: {output_subfolder}")
            plot_dir = os.path.join(OUTPUT_DIR, output_subfolder)
            os.makedirs(plot_dir, exist_ok=True)

        with torch.no_grad():
            batch_size = input_features.shape[0]

            # -----Get initial C* values-----
            c_initial, env_dict = self.preprocess_initial_inputs(x=input_features)

            c_initial = torch.cat([
                c_initial[:, :3],
                input_features[:, 5:13] * c_scaling_factor,
                c_initial[:, 11:]
            ], dim=1)

            # -----Run PINN-----
            c_pred, time_points = self.forward(c_initial=c_initial)
            env_pred = self.process_c_pred(c_pred=c_pred, env_dict=env_dict, time_points=time_points)
            c_pred_at_target = self.get_predictions(c_pred=c_pred, c_initial=c_initial, time_points=time_points)
            
            if debug:
                logger.debug(f"c_pred shape: {c_pred.shape}")
                logger.debug(f"c_pred initial: {c_pred[:, 0, :].detach().cpu().numpy()}")
            
            # -----Store predictions for each sample at all time points-----
            all_predictions = {}
            
            # -----Process each sample for visualization-----
            for batch_idx in range(batch_size):
                full_state = torch.zeros((len(time_points), input_features.shape[1]), dtype=torch.float64, device=self.device)

                full_state[:, 0] = time_points
                full_state[:, 1] = input_features[batch_idx, 1]  # T
                full_state[:, 2] = input_features[batch_idx, 2]  # P
                
                full_state[:, 3] = env_pred[batch_idx, :, 2]
                full_state[:, 4] = env_pred[batch_idx, :, 3]
                
                all_predictions[batch_idx] = {
                    'times': time_points,
                    'predictions': full_state,
                    'c_predictions': c_pred[batch_idx]
                }
                
                if output_subfolder is not None:
                    plot_trajectory(
                        batch_idx=batch_idx,
                        times=time_points,
                        predictions=full_state,
                        c_predictions=c_pred[batch_idx],
                        plot_dir=plot_dir,
                        sample=input_features[batch_idx:batch_idx+1],
                        env_pred=env_pred,
                        env_smoothing_factor=self.env_smoothing_factor
                    )
                    
                    if debug:
                        logger.debug(f"Processed batch {batch_idx}:")
                        logger.debug(f"Target time: {input_features[batch_idx, 0]:.3f}")
                        logger.debug(f"Number of time points: {len(time_points)}")
            
        return c_pred_at_target, all_predictions
