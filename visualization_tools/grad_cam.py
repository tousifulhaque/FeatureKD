"""
Temporal Grad-CAM for Fall Detection Models (CNN and Transformer).

Computes gradient-based importance scores per time step to show
which parts of the time series drive the model's decision.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from argparse import ArgumentParser
import yaml

from Models.fusionTransformer import SingleStreamTransformer
from Models.cnn import SimpleCNN


class TemporalGradCAM:
    """
    Temporal Grad-CAM for time series models (CNN and Transformer).
    
    Computes importance scores per time step by backpropagating
    from the predicted class score through the target layer.
    """
    
    def __init__(self, model, target_layer=None, model_type=None):
        """
        Args:
            model: The model (CNN or Transformer)
            target_layer: Layer to hook (auto-detected if None)
            model_type: 'cnn' or 'transformer' (auto-detected if None)
        """
        self.model = model
        self.activations = None
        self.gradients = None
        self.handles = []
        
        # Auto-detect model type
        if model_type is None:
            self.model_type = self._detect_model_type()
        else:
            self.model_type = model_type
        
        # Auto-select target layer
        if target_layer is None:
            self.target_layer = self._get_default_target_layer()
        else:
            self.target_layer = target_layer
        
        print(f"Grad-CAM initialized: model_type={self.model_type}, target_layer={type(self.target_layer).__name__}")
    
    def _detect_model_type(self):
        """Auto-detect if model is CNN or Transformer."""
        # Check for transformer encoder
        if hasattr(self.model, 'encoder') and isinstance(self.model.encoder, nn.TransformerEncoder):
            return 'transformer'
        # Check for conv layers (CNN)
        elif hasattr(self.model, 'conv4') or hasattr(self.model, 'conv3'):
            return 'cnn'
        else:
            # Default to checking module types
            for module in self.model.modules():
                if isinstance(module, nn.TransformerEncoder):
                    return 'transformer'
            return 'cnn'
    
    def _get_default_target_layer(self):
        """Get default target layer based on model type."""
        if self.model_type == 'transformer':
            # Hook the encoder (outputs [T, B, D])
            return self.model.encoder
        else:
            # CNN: hook last conv layer before pooling
            if hasattr(self.model, 'conv4'):
                return self.model.conv4
            elif hasattr(self.model, 'conv3'):
                return self.model.conv3
            else:
                # Find last Conv1d layer
                last_conv = None
                for module in self.model.modules():
                    if isinstance(module, nn.Conv1d):
                        last_conv = module
                return last_conv
    
    def _forward_hook(self, module, input, output):
        """Store activations during forward pass."""
        if self.model_type == 'transformer':
            # TransformerEncoder output: [T, B, D] -> [B, T, D]
            if output.dim() == 3:
                self.activations = output.permute(1, 0, 2).detach()
            else:
                self.activations = output.detach()
        else:
            # CNN Conv1d output: [B, C, T] -> [B, T, C]
            if output.dim() == 3:
                self.activations = output.permute(0, 2, 1).detach()
            else:
                self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        """Store gradients during backward pass."""
        grad = grad_output[0]
        if self.model_type == 'transformer':
            # TransformerEncoder gradients: [T, B, D] -> [B, T, D]
            if grad.dim() == 3:
                self.gradients = grad.permute(1, 0, 2).detach()
            else:
                self.gradients = grad.detach()
        else:
            # CNN Conv1d gradients: [B, C, T] -> [B, T, C]
            if grad.dim() == 3:
                self.gradients = grad.permute(0, 2, 1).detach()
            else:
                self.gradients = grad.detach()
    
    def register_hooks(self):
        """Register forward and backward hooks."""
        self.handles.append(
            self.target_layer.register_forward_hook(self._forward_hook)
        )
        self.handles.append(
            self.target_layer.register_full_backward_hook(self._backward_hook)
        )
    
    def remove_hooks(self):
        """Remove all hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []
    
    def compute_cam(self, x, target_class=None, method='grad_cam'):
        """
        Compute temporal importance map.
        
        Args:
            x: Input tensor [B, T, C]
            target_class: Class to compute gradients for (None = predicted class)
            method: 'grad_cam' or 'gradient_x_input'
        
        Returns:
            cam: Temporal importance [B, T]
            pred: Predicted class
            prob: Prediction probability
        """
        self.model.eval()
        self.register_hooks()
        
        # Enable gradients for input
        x = x.clone().requires_grad_(True)
        
        # Forward pass
        logits, features = self.model(x)
        
        # Get prediction
        prob = torch.sigmoid(logits).squeeze()
        pred = (prob > 0.5).int()
        
        # Determine target class
        if target_class is None:
            target_class = pred
        
        # For binary classification with single logit:
        # If predicting class 1 (fall), use logit directly
        # If predicting class 0 (ADL), use negative logit
        if target_class == 1:
            score = logits.squeeze()
        else:
            score = -logits.squeeze()
        
        # Backward pass
        self.model.zero_grad()
        score.backward(retain_graph=True)
        
        # Get activations and gradients (both should be [B, T, D] now)
        activations = self.activations
        gradients = self.gradients
        
        # For CNN, we may need to handle downsampled time dimension
        input_seq_len = x.shape[1]
        cam_seq_len = activations.shape[1]
        
        if method == 'grad_cam':
            # Global average pool gradients over time to get channel weights
            weights = gradients.mean(dim=1, keepdim=True)  # [B, 1, D]
            
            # Weighted combination
            cam = (weights * activations).sum(dim=-1)  # [B, T]
        
        elif method == 'gradient_x_input':
            # Gradient × Input (often more stable)
            cam = (gradients * activations).sum(dim=-1)  # [B, T]
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # ReLU to keep positive contributions
        cam = torch.relu(cam)
        
        # Upsample CAM to match input sequence length if needed (for CNN with pooling)
        if cam_seq_len != input_seq_len:
            cam = torch.nn.functional.interpolate(
                cam.unsqueeze(1), 
                size=input_seq_len, 
                mode='linear', 
                align_corners=False
            ).squeeze(1)
        
        # Normalize to [0, 1]
        cam_min = cam.min(dim=1, keepdim=True)[0]
        cam_max = cam.max(dim=1, keepdim=True)[0]
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        
        self.remove_hooks()
        
        return cam.detach(), pred.item(), prob.item()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        
        self.remove_hooks()
        
        return cam.detach(), pred.item(), prob.item()


def cal_smv(sample: torch.Tensor) -> torch.Tensor:
    """Calculate Signal Magnitude Vector."""
    mean = torch.mean(sample, dim=-2, keepdim=True)
    zero_mean = sample - mean
    sum_squared = torch.sum(torch.square(zero_mean), dim=-1, keepdim=True)
    smv = torch.sqrt(sum_squared)
    return torch.cat((sample, smv), dim=-1)


def plot_gradcam_overlay(time_series, cam, sample_idx=0, true_label=None, 
                         pred_label=None, prob=None, feature_names=None,
                         cmap='Reds', save_path=None):
    """
    Plot time series with Grad-CAM overlay.
    
    Args:
        time_series: [B, T, C] or [T, C]
        cam: [B, T] or [T] importance scores
        sample_idx: Which sample to plot
        true_label: Ground truth label
        pred_label: Predicted label
        prob: Prediction probability
        feature_names: Names for each feature
        cmap: Colormap (darker = higher importance)
        save_path: Path to save figure
    """
    # Get data for this sample
    if len(time_series.shape) == 3:
        ts = time_series[sample_idx].cpu().numpy()
        importance = cam[sample_idx].cpu().numpy()
    else:
        ts = time_series.cpu().numpy()
        importance = cam.cpu().numpy()
    
    seq_len, n_features = ts.shape
    
    if feature_names is None:
        feature_names = ['X', 'Y', 'Z', 'SMV'][:n_features]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    
    time_axis = np.arange(seq_len)
    colormap = plt.colormaps.get_cmap(cmap)
    
    # Draw Grad-CAM as rectangular boxes in background
    for t in range(seq_len):
        color = colormap(importance[t])
        ax.axvspan(t, t + 1, facecolor=color, alpha=0.7, edgecolor='none')
    
    # Plot all features
    colors = ['blue', 'green', 'orange', 'purple', 'brown']
    for i in range(n_features):
        label = feature_names[i] if i < len(feature_names) else f'Feature {i}'
        ax.plot(time_axis, ts[:, i], color=colors[i % len(colors)], 
                linewidth=1.5, label=label)
    
    ax.set_xlim(0, seq_len)
    y_margin = 0.1 * (ts.max() - ts.min())
    ax.set_ylim(ts.min() - y_margin, ts.max() + y_margin)
    
    # Add colorbar
    sm = cm.ScalarMappable(cmap=colormap, norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Grad-CAM Importance', fontsize=11)
    
    # Title with prediction info
    if true_label is not None and pred_label is not None:
        true_str = 'Fall' if true_label == 1 else 'ADL'
        pred_str = 'Fall' if pred_label == 1 else 'ADL'
        correct = '✓' if true_label == pred_label else '✗'
        prob_str = f' (p={prob:.2f})' if prob is not None else ''
        title = f'Grad-CAM | True: {true_str} | Pred: {pred_str}{prob_str} {correct}'
    else:
        title = 'Temporal Grad-CAM - Darker = Higher Decision Importance'
    
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Sensor Value', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    return fig


def visualization_gradcam(dataset, labels, model, feature_names=None,
                          cmap='Reds', output_dir='output_gradcam',
                          method='grad_cam', filter='all'):
    """
    Generate Grad-CAM visualizations for all samples.

    Args:
        dataset: [N, T, C] input data
        labels: [N] ground truth labels
        model: The model
        feature_names: Names for features
        cmap: Colormap
        output_dir: Output directory
        method: 'grad_cam' or 'gradient_x_input'
        filter: 'all', 'wrong' (only misclassified), or 'correct' (only correct)
    """
    os.makedirs(output_dir, exist_ok=True)

    if feature_names is None:
        feature_names = ['X', 'Y', 'Z', 'SMV']

    colormap = plt.colormaps.get_cmap(cmap)
    grad_cam = TemporalGradCAM(model)

    saved, skipped = 0, 0
    for i in range(len(dataset)):
        # Get sample
        sample = dataset[i:i+1]
        true_label = int(labels[i].item())

        # Compute Grad-CAM
        cam, pred, prob = grad_cam.compute_cam(sample, method=method)
        importance = cam[0].cpu().numpy()

        # Apply filter
        is_correct = (pred == true_label)
        if filter == 'wrong' and is_correct:
            skipped += 1
            continue
        elif filter == 'correct' and not is_correct:
            skipped += 1
            continue
        
        # Get data
        ts = dataset[i].cpu().numpy()
        seq_len, n_features = ts.shape
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        time_axis = np.arange(seq_len)
        
        # Draw importance rectangles
        for t in range(seq_len):
            color = colormap(importance[t])
            ax.axvspan(t, t + 1, facecolor=color, alpha=0.7, edgecolor='none')
        
        # Plot features
        colors = ['blue', 'green', 'orange', 'purple']
        for j in range(n_features):
            label = feature_names[j] if j < len(feature_names) else f'Feature {j}'
            ax.plot(time_axis, ts[:, j], color=colors[j % len(colors)], 
                    linewidth=1.5, label=label)
        
        ax.set_xlim(0, seq_len)
        y_margin = 0.1 * (ts.max() - ts.min())
        ax.set_ylim(ts.min() - y_margin, ts.max() + y_margin)
        
        # Colorbar
        sm = cm.ScalarMappable(cmap=colormap, norm=Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Importance', fontsize=10)
        
        # Labels
        true_str = 'Fall' if true_label == 1 else 'ADL'
        pred_str = 'Fall' if pred == 1 else 'ADL'
        correct = '✓' if true_label == pred else '✗'
        
        ax.set_xlabel('Time Step', fontsize=11)
        ax.set_ylabel('Sensor Value', fontsize=11)
        ax.set_title(f'Sample {i} | True: {true_str} | Pred: {pred_str} (p={prob:.2f}) {correct}', 
                     fontsize=12)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/GradCAM_{i}.png', dpi=150, bbox_inches='tight')
        plt.close()
        saved += 1

    label_str = {'all': 'all', 'wrong': 'misclassified', 'correct': 'correctly classified'}[filter]
    print(f"Saved {saved} {label_str} Grad-CAM visualizations to {output_dir}/ (skipped {skipped})")


def ablation_test(model, sample, cam, top_k_percent=10):
    """
    Ablation test: zero out top-k% important time steps and check if prediction changes.
    
    Args:
        model: The model
        sample: [1, T, C] input sample
        cam: [1, T] importance scores
        top_k_percent: Percentage of time steps to ablate
    
    Returns:
        original_pred, original_prob, ablated_pred, ablated_prob
    """
    model.eval()
    
    # Original prediction
    with torch.no_grad():
        logits, _ = model(sample)
        original_prob = torch.sigmoid(logits).item()
        original_pred = int(original_prob > 0.5)
    
    # Find top-k% important indices
    importance = cam[0]
    k = max(1, int(len(importance) * top_k_percent / 100))
    top_indices = torch.topk(importance, k).indices
    
    # Ablate (zero out)
    ablated_sample = sample.clone()
    ablated_sample[0, top_indices, :] = 0
    
    # Ablated prediction
    with torch.no_grad():
        logits, _ = model(ablated_sample)
        ablated_prob = torch.sigmoid(logits).item()
        ablated_pred = int(ablated_prob > 0.5)
    
    return original_pred, original_prob, ablated_pred, ablated_prob, top_indices


if __name__ == "__main__":
    # Paths
    sub_id = 29
    weights_path = f"exps/smartfall_fall_wokd/test_data/imu_all_participant_watch_round2_2026-03-24_15-58-52/ttfstudent_{sub_id}.pth"
    data_path = f"data_np/test_{sub_id}.npy"
    
    parser = ArgumentParser(description="Grad-CAM visualization for fall detection")
    parser.add_argument('--config', type=str, default='config/smartfallmm/test.yaml')
    parser.add_argument('--method', type=str, default='grad_cam', 
                        choices=['grad_cam', 'gradient_x_input'],
                        help='Method for computing importance')
    parser.add_argument('--model-type', type=str, default='cnn',
                        choices=['cnn', 'transformer'],
                        help='Model type: cnn or transformer')
    parser.add_argument('--cmap', type=str, default='Reds',
                        help='Colormap (Reds, Greens, Blues, etc.)')
    parser.add_argument('--output-dir', type=str, default=f'visualizations/output_gradcam_{sub_id}',
                        help='Output directory for visualizations')
    parser.add_argument('--sample-idx', type=int, default=None,
                        help='Single sample index to visualize (default: all)')
    parser.add_argument('--ablation', action='store_true',
                        help='Run ablation test on samples')
    parser.add_argument('--filter', type=str, default='wrong',
                        choices=['all', 'wrong', 'correct'],
                        help='Filter samples: all | wrong (misclassified only) | correct (correct only)')
    args = parser.parse_args()
    
    # Load config and model
    with open(args.config, 'r') as f:
        configs = yaml.safe_load(f)
    
    # Load model based on type
    if args.model_type == 'transformer':
        model = SingleStreamTransformer(**configs['model_args'])
    else:
        model = SimpleCNN(**configs['model_args'])
    
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.eval()
    
    # Load data
    data_dict = np.load(data_path, allow_pickle=True).item()
    data = torch.Tensor(data_dict['accelerometer'])
    labels = torch.Tensor(data_dict['labels'])
    imu_data = cal_smv(data)
    
    print(f"Loaded {len(imu_data)} samples")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.sample_idx is not None:
        # Single sample visualization
        sample = imu_data[args.sample_idx:args.sample_idx+1]
        true_label = int(labels[args.sample_idx].item())

        grad_cam = TemporalGradCAM(model)
        cam, pred, prob = grad_cam.compute_cam(sample, method=args.method)

        is_correct = (pred == true_label)
        if args.filter == 'wrong' and is_correct:
            print(f"Sample {args.sample_idx} is correctly predicted — skipped (--filter wrong).")
            exit(0)
        elif args.filter == 'correct' and not is_correct:
            print(f"Sample {args.sample_idx} is misclassified — skipped (--filter correct).")
            exit(0)

        plot_gradcam_overlay(
            time_series=sample,
            cam=cam,
            sample_idx=0,
            true_label=true_label,
            pred_label=pred,
            prob=prob,
            feature_names=['X', 'Y', 'Z', 'SMV'],
            cmap=args.cmap,
            save_path=f'{args.output_dir}/GradCAM_sample_{args.sample_idx}.png'
        )
        
        if args.ablation:
            print("\n--- Ablation Test ---")
            orig_pred, orig_prob, abl_pred, abl_prob, top_idx = ablation_test(
                model, sample, cam, top_k_percent=10
            )
            print(f"Original: {'Fall' if orig_pred else 'ADL'} (p={orig_prob:.3f})")
            print(f"Ablated:  {'Fall' if abl_pred else 'ADL'} (p={abl_prob:.3f})")
            print(f"Prediction changed: {orig_pred != abl_pred}")
            print(f"Top 10% important timesteps: {sorted(top_idx.tolist())}")
    else:
        # All samples
        visualization_gradcam(
            dataset=imu_data,
            labels=labels,
            model=model,
            feature_names=['X', 'Y', 'Z', 'SMV'],
            cmap=args.cmap,
            output_dir=args.output_dir,
            method=args.method,
            filter=args.filter
        )
