# ------------------------------------------------------------------------------------------
# Description: ABLATION MODEL - No time/feature encoding or fusion layers.
#              Raw inputs are directly fed into the main neural network.
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


class PINN(nn.Module):
    def __init__(self, num_inputs=None, num_outputs=None, delta_t=None,
                 device=None, time_batch_size=None, env_smoothing_factor=None, pinn_inputs=None, training_flag=None):
        """
        ABLATION MODEL: No time/feature encoding or fusion layers.
        Raw inputs are directly fed into the main neural network.
        PINN-only mode (no Phase model).

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
        logger.info("ABLATION MODEL: Running without time/feature encoding and fusion layers (PINN-only mode)")

        self.config = {
            'num_inputs': num_inputs,
            'num_outputs': num_outputs,
            'delta_t': delta_t,
            'time_batch_size': time_batch_size,
            'env_smoothing_factor': env_smoothing_factor,
            'pinn_inputs': pinn_inputs,
            'training_flag': training_flag
        }

        # -----Initialize PINN parameters-----
        self.delta_t = delta_t
        self.epsilon = 1e-10
        self.time_batch_size = time_batch_size
        self.env_smoothing_factor = env_smoothing_factor
        self.pinn_inputs = pinn_inputs
        self.training_flag = training_flag

        # -----Neural network architecture (ABLATION: no encoders)-----
        if pinn_inputs == 'time':
            self.num_inputs_pinn = 1
        else:
            self.num_inputs_pinn = 14

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

        self.net.to(self.device)

        # -----Initialize reaction parameters-----
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
        """Override to() method to properly move all components to device"""
        super().to(device)

        self.device = device

        return self

    def preprocess_initial_inputs(self, x):
        """
        Preprocess raw inputs to generate time, pH, and initial concentrations
        PINN-only mode: Uses raw input values directly without Phase model transformation.

        Args:
            x: Raw input features [batch_size, n_features]
               Expected format: [t, T, P, FEED, FH2, XFA, XFB, XFC, XFD, XFE, XFF, XFG, XFI, pH]

        Returns:
            c_initial: tensor with processed inputs for PINN
            env_dict: dictionary with environment parameters
        """
        x = x.to(self.device)
        batch_size = x.shape[0]

        # -----Extract basic features-----
        t = x[:, 0].clone()
        T = x[:, 1]
        P = x[:, 2]
        FEED = x[:, 3]
        FH2 = x[:, 4]

        # -----Extract initial concentrations (XF* values used as C* proxies in PINN-only mode)-----
        CA = x[:, 5]
        CB = x[:, 6]
        CC = x[:, 7]
        CD = x[:, 8]
        CE = x[:, 9]
        CF = x[:, 10]
        CG = x[:, 11]
        CI = x[:, 12]

        pH = x[:, 13]

        FL = torch.zeros(batch_size, dtype=torch.float64, device=self.device)
        VV = torch.ones(batch_size, dtype=torch.float64, device=self.device)
        CS = torch.zeros(batch_size, dtype=torch.float64, device=self.device)

        YA = torch.zeros(batch_size, dtype=torch.float64, device=self.device)
        YB = torch.zeros(batch_size, dtype=torch.float64, device=self.device)
        YC = torch.zeros(batch_size, dtype=torch.float64, device=self.device)
        YD = torch.zeros(batch_size, dtype=torch.float64, device=self.device)
        YE = torch.zeros(batch_size, dtype=torch.float64, device=self.device)
        YF = torch.zeros(batch_size, dtype=torch.float64, device=self.device)
        YI = torch.zeros(batch_size, dtype=torch.float64, device=self.device)

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
        ABLATION: Forward pass using raw inputs directly (no encoding/fusion).
        Predicts C* concentrations at regular time intervals.

        Args:
            c_initial: Input tensor [batch_size, num(t, T, pH, CA, CB, CC, CD, CE, CF, CG, CI, P, FEED, FH2)]
            debug: Whether to print debug information

        Returns:
            c_pred: Predicted C* concentrations [batch_size, n_time_points, 8]
            time_points: Tensor of time points [n_time_points]
        """
        batch_size = c_initial.shape[0]

        # -----Extract sample-specific features once (ABLATION: raw features, no encoding)-----
        sample_features = torch.cat([
            c_initial[:, 1:2],
            c_initial[:, 2:3],
            c_initial[:, 3:11],
            c_initial[:, 11:14],
        ], dim=1)

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

                if self.pinn_inputs == 'time':
                    species_outputs = self.net(t_batch_tensor)
                elif self.pinn_inputs == 'time+initials':
                    combined_features = torch.cat([t_batch_tensor, sample_features], dim=1)
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

        # -----Initialize storage for results using lists instead of tensors-----
        x_pred_list = []
        env_pred_list = []    

        # -----Set initial x values (t=0)-----
        Y_values_initial = torch.stack([YA_initial, YB_initial, YC_initial, YD_initial, YE_initial, YF_initial, YI_initial], dim=1)
        _, __, x_initial = self.convert_c_to_x(
            c_pred[:, 0], FL_initial, VV_initial, CS_initial, FH2_initial, Y_values_initial
        )

        # -----Add initial values to lists-----
        x_pred_list.append(x_initial)

        # -----Initial environment state - create a copy to avoid in-place modification-----
        env_initial = torch.stack([T_initial, P_initial, FEED_initial, FH2_initial,
                                   pH_initial, FL_initial, VV_initial, CS_initial,
                                   YA_initial, YB_initial, YC_initial, YD_initial,
                                   YE_initial, YF_initial, YI_initial], dim=1) 

        env_pred_list.append(env_initial)

        # -----For each time point (except t=0), predict concentrations-----
        for t_idx in range(1, n_time_points):
            with torch.enable_grad():
                # -----Get current environment parameters from previous step-----
                current_env = env_pred_list[-1].clone()

                # -----Extract values from current environment-----
                current_T = current_env[:, 0]
                current_P = current_env[:, 1]
                current_FH2 = current_env[:, 3]
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

                # -----Predict XF* concentrations-----
                Y_values = torch.stack([current_YA, current_YB, current_YC, current_YD, current_YE, current_YF, current_YI], dim=1)
                FEED_current, FH2_current, x_current = self.convert_c_to_x(c_pred[:, t_idx], current_FL, current_VV, current_CS, current_FH2, Y_values)

                x_pred_list.append(x_current)

                # -----PINN-only mode: Keep environment parameters constant-----
                env_new = torch.stack([
                    current_T, current_P, FEED_current, FH2_current,
                    pH_initial, FL_initial, VV_initial, CS_initial,
                    YA_initial, YB_initial, YC_initial, YD_initial,
                    YE_initial, YF_initial, YI_initial
                ], dim=1)

                env_pred_list.append(env_new)

        # -----Create the final tensors by stacking the lists-----
        x_pred = torch.stack(x_pred_list, dim=1)
        env_pred = torch.stack(env_pred_list, dim=1)

        return x_pred, env_pred
    
    def get_predictions(self, x_pred=None, c_pred=None, c_initial=None, time_points=None, env_pred=None):
        """
        Get predictions for both x_pred and c_pred based on target time in c_initial
        
        Args:
            x_pred: Predicted x values [batch_size, n_time_points, 8]
            c_pred: Predicted C* concentrations [batch_size, n_time_points, 8]
            c_initial: Initial concentrations [batch_size, n_features]
            time_points: Tensor of time points [n_time_points]
            env_pred: Environment parameter predictions [batch_size, n_time_points, n_env_vars]
        Returns:
            u_pred: Predicted u values [batch_size, 13]
            c_pred_at_target: Predicted C* values at target times [batch_size, 8]
        """
        batch_size = c_initial.shape[0]

        # -----Extract target times from c_initial-----
        target_times = c_initial[:, 0]
        
        # -----Initialize output tensors-----
        u_pred = torch.zeros((batch_size, 13), dtype=torch.float64, device=self.device)
        c_pred_at_target = torch.zeros((batch_size, c_pred.shape[-1]), dtype=torch.float64, device=self.device)
        
        # -----Process each sample independently-----
        for i in range(batch_size):
            t_target = target_times[i]
            
            time_diffs = torch.abs(time_points - t_target)
            closest_idx = torch.argmin(time_diffs).item()

            if time_points[closest_idx] > t_target and closest_idx > 0:
                closest_idx -= 1
            
            u_pred[i, 0] = time_points[closest_idx]
            u_pred[i, 1] = env_pred[i, closest_idx, 0]
            u_pred[i, 2] = env_pred[i, closest_idx, 1]
            u_pred[i, 3] = env_pred[i, closest_idx, 2]
            u_pred[i, 4] = env_pred[i, closest_idx, 3]
            u_pred[i, 5:13] = x_pred[i, closest_idx]
            
            c_pred_at_target[i] = c_pred[i, closest_idx]
        
        return u_pred, c_pred_at_target

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
        Load saved PINN model (PINN-only mode)

        Args:
            model_path: path to saved model file
        Returns:
            loaded PINN model
        """
        # -----Determine the appropriate device-----
        map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # -----Load the checkpoint with device mapping-----
        checkpoint = torch.load(model_path, map_location=map_location)

        # Get the device object if string was provided
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

            # -----Preprocess inputs to get initial C* values (PINN-only mode)-----
            c_initial, env_dict = self.preprocess_initial_inputs(x=input_features)

            # -----Replace C* concentrations with XF* concentrations-----
            c_initial = torch.cat([
                c_initial[:, :3],
                input_features[:, 5:13] * c_scaling_factor,
                c_initial[:, 11:]
            ], dim=1)

            # -----Run PINN-----
            c_pred, time_points = self.forward(c_initial=c_initial)
            x_pred, env_pred = self.process_c_pred(c_pred=c_pred, env_dict=env_dict, time_points=time_points)
            final_pred, c_pred_at_target = self.get_predictions(x_pred=x_pred, c_pred=c_pred, c_initial=c_initial, 
                                                                time_points=time_points, env_pred=env_pred)
            
            if debug:
                logger.debug(f"c_pred shape: {c_pred.shape}")
                logger.debug(f"c_pred initial: {c_pred[:, 0, :].detach().cpu().numpy()}")
            
            # -----Store predictions for each sample at all time points-----
            all_predictions = {}
            
            # -----Process each sample for visualization-----
            for batch_idx in range(batch_size):
                # Create full state tensor for this sample
                full_state = torch.zeros((len(time_points), input_features.shape[1]), dtype=torch.float64, device=self.device)

                full_state[:, 0] = time_points
                full_state[:, 1] = input_features[batch_idx, 1]
                full_state[:, 2] = input_features[batch_idx, 2]
                full_state[:, 3] = env_pred[batch_idx, :, 2]
                full_state[:, 4] = env_pred[batch_idx, :, 3]
                full_state[:, 5:13] = x_pred[batch_idx]
                
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
                        env_smoothing_factor=self.env_smoothing_factor,
                        run_mode='pinn'
                    )
                    
                    if debug:
                        logger.debug(f"Processed batch {batch_idx}:")
                        logger.debug(f"Target time: {input_features[batch_idx, 0]:.3f}")
                        logger.debug(f"Number of time points: {len(time_points)}")
            
        return final_pred, c_pred_at_target, all_predictions