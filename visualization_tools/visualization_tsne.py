"""
t-SNE Visualization for Fall Detection Models (CNN and Transformer).

Visualizes feature embeddings from different model architectures.
"""

import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch 
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from argparse import ArgumentParser
import yaml

from Models.fusionTransformer import SingleStreamTransformer
from Models.tcn import SimpleCNN


def cal_smv(sample: torch.Tensor) -> torch.Tensor:
    """Calculate Signal Magnitude Vector."""
    mean = torch.mean(sample, dim=-2, keepdim=True)
    zero_mean = sample - mean
    sum_squared = torch.sum(torch.square(zero_mean), dim=-1, keepdim=True) 
    smv = torch.sqrt(sum_squared)
    return torch.cat((sample, smv), dim=-1)


def extract_features(model, data, model_type='cnn'):
    """
    Extract features from model.
    
    Args:
        model: The model (CNN or Transformer)
        data: Input data [N, T, C]
        model_type: 'cnn' or 'transformer'
    
    Returns:
        features: [N, D] flattened feature vectors
    """
    model.eval()
    with torch.no_grad():
        _, features = model(data)
    
    # Flatten features for t-SNE
    if features.dim() == 3:
        # [N, T, D] -> [N, T*D]
        features = features.reshape(features.shape[0], -1)
    elif features.dim() == 2:
        # [N, D] - already flat
        pass
    
    return features.cpu().numpy()


def plot_tsne(features, labels, output_path, title="t-SNE Visualization", 
              perplexity=30, random_state=42):
    """
    Plot t-SNE visualization.
    
    Args:
        features: [N, D] feature vectors
        labels: [N] labels
        output_path: Path to save figure
        title: Plot title
        perplexity: t-SNE perplexity
        random_state: Random seed
    """
    # t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate="auto", 
                init="pca", random_state=random_state)
    X2 = tsne.fit_transform(features)
    
    # Plot by class
    plt.figure(figsize=(10, 8))
    classes = np.unique(labels)
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    class_names = {0: 'ADL', 1: 'Fall'}
    
    for i, cls in enumerate(classes):
        idx = labels == cls
        label_name = class_names.get(int(cls), f"Class {int(cls)}")
        plt.scatter(X2[idx, 0], X2[idx, 1], s=15, alpha=0.7, 
                    color=colors[i % len(colors)], label=label_name)
    
    plt.legend(markerscale=2, fontsize=12)
    plt.title(title, fontsize=14)
    plt.xlabel("t-SNE 1", fontsize=12)
    plt.ylabel("t-SNE 2", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = ArgumentParser(description="t-SNE visualization for fall detection models")
    parser.add_argument('--config', type=str, default='../../config/smartfallmm/test.yaml',
                        help='Path to config file')
    parser.add_argument('--weights', type=str, 
                        default='../../exps/smartfall_fall_wokd/test_data/imu_all_participant_watch_cnn_round2/ttfstudent_44.pth',
                        help='Path to model weights')
    parser.add_argument('--data', type=str, default='../../data_np/test_44.npy',
                        help='Path to data file')
    parser.add_argument('--model-type', type=str, default='cnn',
                        choices=['cnn', 'transformer'],
                        help='Model type: cnn or transformer')
    parser.add_argument('--output-dir', type=str, default='tsne_visualization',
                        help='Output directory')
    parser.add_argument('--perplexity', type=int, default=30,
                        help='t-SNE perplexity')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load config
    with open(args.config, 'r') as f:
        configs = yaml.safe_load(f)
    
    # Load model based on type
    if args.model_type == 'transformer':
        model = SingleStreamTransformer(**configs['model_args'])
    else:
        model = SimpleCNN(**configs['model_args'])
    
    model.load_state_dict(torch.load(args.weights, weights_only=True))
    model.eval()
    print(f"Loaded {args.model_type.upper()} model from {args.weights}")
    
    # Load data
    data_dict = np.load(args.data, allow_pickle=True).item()
    data = torch.Tensor(data_dict['accelerometer'])
    labels = np.array(data_dict['labels'])
    
    # Add SMV
    imu_data = cal_smv(data)
    print(f"Loaded {len(imu_data)} samples, shape: {imu_data.shape}")
    
    # Extract features
    features = extract_features(model, imu_data, model_type=args.model_type)
    print(f"Extracted features shape: {features.shape}")
    
    # Plot t-SNE
    output_path = f"{args.output_dir}/tsne_{args.model_type}.png"
    plot_tsne(
        features=features,
        labels=labels,
        output_path=output_path,
        title=f"t-SNE: {args.model_type.upper()} Features by Label",
        perplexity=args.perplexity
    )