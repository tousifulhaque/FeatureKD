import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class AttentionExtractor:
    """Extract attention weights from transformer encoder layers."""
    
    def __init__(self, model):
        self.model = model
        self.attention_weights = []
        self.hooks = []
    
    def register_hooks(self):
        """Register forward hooks on all self-attention layers."""
        self.attention_weights = []
        
        def pre_hook_fn(module, args):
            # Modify forward call to enable attention weights
            # args = (query, key, value, ...)
            # We need to set need_weights=True in kwargs
            return args
        
        def hook_fn(module, input, output):
            # For MultiheadAttention, output is (attn_output, attn_weights)
            if isinstance(output, tuple) and len(output) == 2 and output[1] is not None:
                self.attention_weights.append(output[1].detach().cpu())
        
        # Find all MultiheadAttention modules
        for name, module in self.model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                # Patch the module to always return attention weights
                original_forward = module.forward
                def make_patched_forward(orig_fwd):
                    def patched_forward(query, key, value, **kwargs):
                        kwargs['need_weights'] = True
                        kwargs['average_attn_weights'] = False
                        return orig_fwd(query, key, value, **kwargs)
                    return patched_forward
                module.forward = make_patched_forward(original_forward)
                hook = module.register_forward_hook(hook_fn)
                self.hooks.append((hook, module, original_forward))
    
    def remove_hooks(self):
        """Remove all registered hooks and restore original forward methods."""
        for hook, module, original_forward in self.hooks:
            hook.remove()
            module.forward = original_forward
        self.hooks = []
    
    def get_attention(self, x):
        """Forward pass and return attention weights."""
        self.attention_weights = []
        self.model.eval()
        with torch.no_grad():
            _ = self.model(x)
        return self.attention_weights
