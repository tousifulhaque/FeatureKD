import torch
import torch.nn as nn
import torch.nn.functional as F

# Simple model
encoder = nn.Linear(10, 5)
mlp = nn.Linear(5, 2)

# Create optimizer
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(mlp.parameters()))

x = torch.randn(4, 10)
target_features = torch.randn(4, 5)
labels = torch.randint(0, 2, (4,))

# Define loss functions
def compute_encoder_loss(features):
    return F.mse_loss(features, target_features)

def compute_mlp_loss(logits, labels):
    return F.cross_entropy(logits, labels)

# Forward
features = encoder(x)
logits = mlp(features)

encoder_loss = compute_encoder_loss(features)
mlp_loss = compute_mlp_loss(logits, labels)

# Method 1: Check encoder_loss contribution to MLP
optimizer.zero_grad()
encoder_loss.backward(retain_graph=True)
print("MLP gradients from encoder_loss:")
for name, param in mlp.named_parameters():
    if param.grad is not None:
        print(f"  {name}: {param.grad.norm().item():.8f}")  # Should be 0 or None
    else:
        print(f"  {name}: None")

# Reset and check total
optimizer.zero_grad()
features = encoder(x)  # Need to recompute forward pass
logits = mlp(features)
encoder_loss = compute_encoder_loss(features)
mlp_loss = compute_mlp_loss(logits, labels)

mlp_loss.backward(retain_graph=True)  # Need retain_graph since we'll use the graph again
print("\nMLP gradient for mlp_loss:")
for name, param in mlp.named_parameters():
    if param.grad is not None: 
        print(f"{name} : {param.grad.norm().item():.6f}")
    else: 
        print(f"{name}: None")

# Need to recompute forward pass for clean graph
optimizer.zero_grad()
features = encoder(x)
logits = mlp(features)
encoder_loss = compute_encoder_loss(features)
mlp_loss = compute_mlp_loss(logits, labels)
total_loss = encoder_loss + mlp_loss
total_loss.backward()

print("\nMLP gradients from total_loss:")
for name, param in mlp.named_parameters():
    if param.grad is not None:
        print(f"  {name}: {param.grad.norm().item():.6f}")  # Only from mlp_loss
    else:
        print(f"  {name}: None")
