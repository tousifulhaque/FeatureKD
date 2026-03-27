import os
import torch 
from argparse import ArgumentParser
import yaml
import numpy as np 
from Models.fusionTransformer import SingleStreamTransformer
import matplotlib.pyplot as plt
from utils.attention_hook import AttentionExtractor

from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm

def plot_attention_overlay(time_series, attention_weights, sample_idx=0, 
                           layer_idx=-1, head_idx=None, save_path=None, 
                           feature_names=None, cmap='Greens'):
    """
    Overlay attention weights on time series data as a single plot with all features.
    Attention shown as rectangular boxes - darker = higher attention.
    
    Args:
        time_series: (batch, seq_len, features) or (seq_len, features)
        attention_weights: list of (batch, heads, seq_len, seq_len) per layer
        sample_idx: which sample in batch to visualize
        layer_idx: which layer's attention (-1 for last layer)
        head_idx: which head (None = average all heads)
        save_path: path to save figure
        feature_names: list of names for features (default: ['X', 'Y', 'Z', 'SMV'])
        cmap: colormap name (darker = higher attention)
    """
    # Get time series for this sample
    if len(time_series.shape) == 3:
        ts = time_series[sample_idx].cpu().numpy()
    else:
        ts = time_series.cpu().numpy()
    
    seq_len, n_features = ts.shape
    
    # Default feature names
    if feature_names is None:
        feature_names = ['X', 'Y', 'Z', 'SMV'][:n_features]
    
    # Get attention for specified layer
    attn = attention_weights[layer_idx][sample_idx]  # (heads, seq_len, seq_len)
    
    # Average over heads or select specific head
    if head_idx is None:
        attn = attn.mean(dim=0).numpy()  # (seq_len, seq_len)
    else:
        attn = attn[head_idx].numpy()
    
    # Aggregate attention: how much attention each timestep receives
    attn_scores = attn.sum(axis=0)  # (seq_len,)
    
    # Normalize to [0, 1]
    attn_scores = (attn_scores - attn_scores.min()) / (attn_scores.max() - attn_scores.min() + 1e-8)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    
    time_axis = np.arange(seq_len)
    
    # Get colormap
    colormap = plt.colormaps.get_cmap(cmap)
    
    # Get y-axis limits from all features
    y_min = ts.min() - 0.1 * np.abs(ts.min())
    y_max = ts.max() + 0.1 * np.abs(ts.max())
    
    # Draw attention as rectangular boxes in background
    for t in range(seq_len):
        color = colormap(attn_scores[t])
        ax.axvspan(t, t + 1, facecolor=color, alpha=0.6, edgecolor='none')
    
    # Plot all features on top
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink']
    for i in range(n_features):
        ax.plot(time_axis, ts[:, i], color=colors[i % len(colors)], 
                linewidth=1.5, label=feature_names[i] if i < len(feature_names) else f'Feature {i}')
    
    ax.set_xlim(0, seq_len)
    ax.set_ylim(y_min, y_max)
    
    # Add colorbar for attention
    sm = cm.ScalarMappable(cmap=colormap, norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Attention Score', fontsize=12)
    
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Sensor Value', fontsize=12)
    ax.set_title(f'Attention Overlay (Layer {layer_idx}) - Darker = Higher Attention', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    return fig


def plot_attention_heatmap(attention_weights, sample_idx=0, layer_idx=-1,
                           head_idx=None, save_path=None):
    """
    Plot attention matrix as heatmap.
    
    Args:
        attention_weights: list of (batch, heads, seq_len, seq_len) per layer
        sample_idx: which sample in batch
        layer_idx: which layer
        head_idx: which head (None = average all)
        save_path: path to save figure
    """
    attn = attention_weights[layer_idx][sample_idx]
    
    if head_idx is None:
        attn = attn.mean(dim=0).numpy()
        title = f'Average Attention (Layer {layer_idx})'
    else:
        attn = attn[head_idx].numpy()
        title = f'Attention Head {head_idx} (Layer {layer_idx})'
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(attn, cmap='hot', aspect='auto')
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Attention Weight')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    return fig
def cal_smv(sample: torch.Tensor) -> torch.Tensor:
    '''
    Function to calculate SMV
    '''
    mean = torch.mean(sample, dim=-2, keepdim=True)
    zero_mean = sample - mean
    sum_squared = torch.sum(torch.square(zero_mean), dim=-1, keepdim=True) 
    smv = torch.sqrt(sum_squared)
    return torch.cat((sample, smv), dim=-1)


def visualization_inference(dataset, predictions, labels, attention_weights, 
                            model, extractor, feature_names=None, cmap='Greens'):
    """
    Visualize inference results with attention overlay.
    
    Args:
        dataset: (N, seq_len, features) tensor of input data
        predictions: (N,) predicted labels
        labels: (N,) ground truth labels
        attention_weights: pre-computed attention weights or None to compute per sample
        model: the model (used if attention_weights is None)
        extractor: AttentionExtractor instance
        feature_names: list of feature names
        cmap: colormap for attention
    """
    visualization_dir = "output_visualization"
    os.makedirs(visualization_dir, exist_ok=True)
    
    if feature_names is None:
        feature_names = ['X', 'Y', 'Z', 'SMV']
    
    colormap = plt.colormaps.get_cmap(cmap)
    
    for i in range(len(dataset)):
        # Get attention for this sample
        sample_data = dataset[i:i+1]
        extractor.register_hooks()
        attn_weights = extractor.get_attention(sample_data)
        extractor.remove_hooks()
        
        # Get attention scores
        attn = attn_weights[-1][0]  # last layer, first (only) sample
        attn = attn.mean(dim=0).numpy()  # average over heads
        attn_scores = attn.sum(axis=0)  # aggregate attention per timestep
        attn_scores = (attn_scores - attn_scores.min()) / (attn_scores.max() - attn_scores.min() + 1e-8)
        
        # Get data
        ts = dataset[i].cpu().numpy()
        seq_len, n_features = ts.shape
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Draw attention rectangles in background
        for t in range(seq_len):
            color = colormap(attn_scores[t])
            ax.axvspan(t, t + 1, facecolor=color, alpha=0.6, edgecolor='none')
        
        # Plot all features
        time_axis = np.arange(seq_len)
        colors = ['blue', 'green', 'orange', 'purple']
        for j in range(n_features):
            label = feature_names[j] if j < len(feature_names) else f'Feature {j}'
            ax.plot(time_axis, ts[:, j], color=colors[j % len(colors)], 
                    linewidth=1.5, label=label)
        
        ax.set_xlim(0, seq_len)
        ax.set_ylim(ts.min() - 0.1 * np.abs(ts.min()), ts.max() + 0.1 * np.abs(ts.max()))
        
        # Add colorbar
        sm = cm.ScalarMappable(cmap=colormap, norm=Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Attention', fontsize=10)
        
        # Labels
        true_label = 'Fall' if labels[i] == 1 else 'ADL'
        pred_label = 'Fall' if predictions[i] == 1 else 'ADL'
        correct = '✓' if labels[i] == predictions[i] else '✗'
        
        ax.set_xlabel('Time Step', fontsize=11)
        ax.set_ylabel('Sensor Value', fontsize=11)
        ax.set_title(f'Sample {i} | True: {true_label} | Pred: {pred_label} {correct}', fontsize=12)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(f'{visualization_dir}/Visualization_{i}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Saved {len(dataset)} visualizations to {visualization_dir}/")


if __name__ == "__main__":
    weights_path = "../../exps/smartfall_fall_wokd/test_data/imu_all_participant_watch/ttfstudent_61.pth"
    data_path = "../../data_np/test_61.npy"
    parser = ArgumentParser(usage="Gets configuration of model")
    parser.add_argument('--config', type=str, default='../../config/smartfallmm/student.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        configs = yaml.safe_load(file)
    model = SingleStreamTransformer(**configs['model_args'])
    model.eval()
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    data = torch.Tensor(np.load(data_path, allow_pickle=True).item()['accelerometer'])
    labels = torch.Tensor(np.load(data_path, allow_pickle=True).item()['labels'])
    imu_data = cal_smv(data)
    extractor = AttentionExtractor(model)
    
    # Run inference
    with torch.no_grad():
        outputs = torch.sigmoid(model(imu_data)[0]) > 0.5
        predictions = outputs.int().squeeze()
    
    # Visualize all samples with attention overlay
    visualization_inference(
        dataset=imu_data,
        predictions=predictions,
        labels=labels,
        attention_weights=None,  # computed per sample inside
        model=model,
        extractor=extractor,
        feature_names=['X', 'Y', 'Z', 'SMV'],
        cmap='Greens'
    )

