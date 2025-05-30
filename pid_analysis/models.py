import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import itertools
from typing import List, Callable

# Import AMP functionality
try:
    from torch.amp import autocast, GradScaler
    
    class DummyAMPModule:
        def __init__(self):
            self.autocast = autocast
            self.GradScaler = GradScaler
    
    amp = DummyAMPModule()
    
except ImportError:
    print("Warning: torch.amp not available. Mixed precision training disabled.")
    
    class DummyAMPModule:
        def __init__(self):
            pass
        
        def autocast(self, *args, **kwargs):
            from contextlib import nullcontext
            return nullcontext()
        
        def GradScaler(self, device='cuda', *args, **kwargs):
            class DummyScaler:
                def scale(self, loss): return loss
                def step(self, optimizer): optimizer.step()
                def update(self): pass
            return DummyScaler()
    
    amp = DummyAMPModule()

from .sinkhorn import sinkhorn_probs

# Global configurations (avoid circular imports by defining locally)
global_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = False  # Default value
PRECISION = torch.float16  # Default precision for AMP
CHUNK_SIZE = 128  # Default chunk size
AGGRESSIVE_CLEANUP = False  # Whether to aggressively clean memory
MEMORY_CLEANUP_INTERVAL = 10  # Clean memory every N chunks

# MLP and simple_discrim use the global_device upon creation of layers/tensors

def mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    layers: int,
    activation: str
) -> nn.Sequential:
    act_map = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh
    }
    if activation not in act_map:
        raise ValueError(f"Unsupported activation: {activation}. Choose from {list(act_map.keys())}")
    act_layer = act_map[activation]
    modules = [
        nn.Linear(input_dim, hidden_dim),
        act_layer()
    ]
    for _ in range(layers):
        modules.extend([
            nn.Linear(hidden_dim, hidden_dim),
            act_layer()
        ])
    modules.append(nn.Linear(hidden_dim, output_dim))
    # Layers are moved to device when the model containing the mlp is moved.
    # Or, ensure .to(global_device) is called if mlp is used standalone.
    return nn.Sequential(*modules) #.to(global_device) -> model that uses it will be moved to device

def simple_discrim(
    xs: List[torch.Tensor],
    y: torch.Tensor,
    num_labels: int
) -> Callable[[torch.Tensor, ...], torch.Tensor]:
    shape = [x.size(1) for x in xs] + [num_labels]
    # Ensure tensors are on the correct global device
    p = torch.ones(*shape, device=global_device) * 1e-8 
    # y is expected to be on global_device by the caller
    # xs elements are expected to be on global_device by the caller
    for i in range(y.size(0)):
        input_indices = [torch.argmax(x[i]).item() for x in xs]
        indices = input_indices + [y[i].item()]
        p[tuple(indices)] += 1
    
    input_shapes = [x.size(1) for x in xs]
    input_indices_list = list(itertools.product(*[range(s) for s in input_shapes]))
    
    for input_indices in input_indices_list:
        idx = list(input_indices)
        marginal_sum = p[tuple(idx + [slice(None)])].sum()
        if marginal_sum > 0:
            p[tuple(idx + [slice(None)])] /= marginal_sum
    
    def discriminator_function(*inputs: torch.Tensor) -> torch.Tensor:
        # inputs are expected to be on global_device
        indices = [torch.argmax(inp, dim=1) for inp in inputs]
        return torch.log(p[tuple(indices)])
    return discriminator_function

class Discrim(nn.Module):
    def __init__(
        self,
        x_dim: int,
        hidden_dim: int,
        num_labels: int,
        layers: int,
        activation: str
    ):
        super().__init__()
        # mlp will be on CPU initially, then moved to device when Discrim instance is moved
        self.mlp = mlp(x_dim, hidden_dim, num_labels, layers, activation)
        
    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        # xs are expected to be on the same device as the model
        x = torch.cat(xs, dim=-1)
        return self.mlp(x)

class CEAlignment(nn.Module):
    def __init__(
        self,
        x1_dim: int,
        x2_dim: int,
        hidden_dim: int,
        embed_dim: int,
        num_labels: int,
        layers: int,
        activation: str
    ):
        super().__init__()
        self.num_labels = num_labels
        # mlp1 and mlp2 will be on CPU initially, then moved to device with CEAlignment instance
        self.mlp1 = mlp(x1_dim, hidden_dim, embed_dim * num_labels, layers, activation)
        self.mlp2 = mlp(x2_dim, hidden_dim, embed_dim * num_labels, layers, activation)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        p_y_x1: torch.Tensor,
        p_y_x2: torch.Tensor
    ) -> torch.Tensor:
        batch_size = x1.size(0)
        if x2.size(0) != batch_size:
            # Ensure consistent batch sizes for x2 and p_y_x2 if x1 drives batch size
            # This assumes x1, p_y_x1 are primary and x2, p_y_x2 might be from a different source batch
            x2 = x2[:batch_size]
            p_y_x2 = p_y_x2[:batch_size]
        
        q1 = self.mlp1(x1).unflatten(1, (self.num_labels, -1))
        q2 = self.mlp2(x2).unflatten(1, (self.num_labels, -1))

        q1 = (q1 - q1.mean(dim=2, keepdim=True)) / torch.sqrt(q1.var(dim=2, keepdim=True) + 1e-8)
        q2 = (q2 - q2.mean(dim=2, keepdim=True)) / torch.sqrt(q2.var(dim=2, keepdim=True) + 1e-8)

        aff = torch.einsum('bce, dce -> bdc', q1, q2) / math.sqrt(q1.size(-1))
        aff_max = aff.reshape(-1, aff.size(-1)).max(dim=0)[0]
        aff_centered = aff - aff_max.view(1, 1, -1)
        A = torch.exp(aff_centered)

        couplings = []
        for c in range(self.num_labels):
            # PAPER'S APPROACH: Use separate marginals directly as per Algorithm 1
            # Row marginal = p(y=c | x1), Column marginal = p(y=c | x2)
            coupling_c = sinkhorn_probs(
                A[..., c],
                p_y_x1[:, c],  # row-marginal = p(y=c | x1) - no averaging
                p_y_x2[:, c],  # col-marginal = p(y=c | x2) - no averaging
                # Potentially pass chunk_size from this model if needed by sinkhorn
                # chunk_size = self.chunk_size # If CEAlignment had self.chunk_size
            )
            couplings.append(coupling_c)
        
        P = torch.stack(couplings, dim=-1)
        # P = P / (P.sum() + 1e-8) # Normalize P across all batch, batch, labels - REMOVED to preserve exact marginal constraints from Sinkhorn
        return P

class CEAlignmentInformation(nn.Module):
    def __init__(self, x1_dim, x2_dim, hidden_dim, embed_dim, num_labels,
                 layers, activation, discrim_1, discrim_2, discrim_12, p_y):
        super().__init__()
        self.num_labels = num_labels
        # CEAlignment will be on CPU, then moved to global_device with CEAlignmentInformation instance
        self.align = CEAlignment(x1_dim, x2_dim, hidden_dim, embed_dim, num_labels, layers, activation)
        
        # Discriminators are expected to be already on the correct device
        self.discrim_1 = discrim_1
        self.discrim_2 = discrim_2
        self.discrim_12 = discrim_12
        
        for D_module in (self.discrim_1, self.discrim_2, self.discrim_12):
            if isinstance(D_module, nn.Module): # Check if it's an nn.Module before calling eval/parameters
                D_module.eval()
                for param in D_module.parameters():
                    param.requires_grad = False
            # If simple_discrim (callable), it doesn't have eval() or parameters()
                
        self.register_buffer('p_y', p_y) # p_y should be on the correct device when passed
        
        # Use global AMP settings
        self.use_amp = USE_AMP
        if self.use_amp:
            # Note: GradScaler is typically used in the training loop, not as a model attribute,
            # unless the model itself performs the optimizer step.
            # For now, we assume the training loop will handle the scaler.
            # If this model were to call optimizer.step() internally, it would need its own scaler.
            self.scaler = amp.GradScaler() # GradScaler doesn't take device as parameter
        else:
            self.scaler = None # Or a dummy scaler that does nothing
        
        # Use global config values
        self.chunk_size = CHUNK_SIZE
        self.aggressive_cleanup = AGGRESSIVE_CLEANUP
        self.memory_cleanup_interval = MEMORY_CLEANUP_INTERVAL

    def forward(self, x1, x2, y): # y is for an optional loss component if CE itself was trained with supervised loss
        # x1, x2, y are expected to be on the model's device
        # device_type for autocast
        current_device_type = x1.device.type # 'cuda' or 'cpu'
        autocast_dtype = PRECISION if current_device_type == 'cuda' else torch.bfloat16

        with amp.autocast(device_type=current_device_type, dtype=autocast_dtype, enabled=self.use_amp):
            if self.discrim_1 is None:
                raise ValueError("discrim_1 cannot be None. Must provide trained discriminators.")
            if self.discrim_2 is None:
                raise ValueError("discrim_2 cannot be None. Must provide trained discriminators.")
            if self.discrim_12 is None:
                raise ValueError("discrim_12 cannot be None. Must provide trained discriminators.")
                
            p_y_x1 = F.softmax(self.discrim_1(x1), dim=1)
            p_y_x2 = F.softmax(self.discrim_2(x2), dim=1)
            p_y_x1x2 = F.softmax(self.discrim_12(torch.cat([x1, x2], dim=-1)), dim=1)
        
        # Calculations outside autocast if they need full precision or cause issues with AMP
        # However, these ops are generally AMP-safe.
        # Ensure p_y is on the same device as p_y_x1 for the division.
        p_y_on_device = self.p_y.to(p_y_x1.device)

        mi_x1_y = torch.sum(p_y_x1 * torch.log(p_y_x1 / p_y_on_device.unsqueeze(0) + 1e-8), dim=1, dtype=torch.float32)
        mi_x2_y = torch.sum(p_y_x2 * torch.log(p_y_x2 / p_y_on_device.unsqueeze(0) + 1e-8), dim=1, dtype=torch.float32)
        
        # The align.forward itself might use sinkhorn_probs which respects global AMP settings
        # No explicit autocast here needed if align.forward and sinkhorn are AMP-aware or handle it internally.
        P = self.align(x1, x2, p_y_x1, p_y_x2)
        
        P_cond = P / (P.sum(dim=-1, keepdim=True) + 1e-8)
        p_y_expanded = p_y_on_device.view(1, 1, -1)
        log_ratio = torch.log(P_cond + 1e-8) - torch.log(p_y_expanded + 1e-8)
        mi_x1x2_y = (P * log_ratio).sum(dim=[1, 2], dtype=torch.float32) # Sum over batch and labels for each sample in primary batch dim
        
        mi_discrim_x1x2_y = torch.sum(p_y_x1x2 * torch.log(p_y_x1x2 / p_y_on_device.unsqueeze(0) + 1e-8), 
                                     dim=1, dtype=torch.float32)
        
        # PID Components - ensure these are per-sample before .mean()
        redundancy_samples = torch.clamp(mi_x1_y + mi_x2_y - mi_x1x2_y, min=0)
        unique1_samples = torch.clamp(mi_x1_y - redundancy_samples, min=0)
        unique2_samples = torch.clamp(mi_x2_y - redundancy_samples, min=0)
        
        # Synergy calculation based on paper's I_p and I_q formulation
        # I_p(X1,X2;Y) is from the joint discriminator (p_y_x1x2)
        # I_q(X1,X2;Y) is from the alignment model (mi_x1x2_y from P)
        synergy_samples = torch.clamp(mi_discrim_x1x2_y - mi_x1x2_y, min=0)
        
        # Loss for optimizing the alignment model (self.align) is typically to maximize I_q(X1,X2;Y)
        # or minimize KL divergence related to it.
        # Here, loss is -I_q(X1,X2;Y).mean() as we want to maximize I_q.
        loss = -mi_x1x2_y.mean() 
        
        # Average PID values over the batch for reporting/logging
        pid_vals = torch.stack([
            redundancy_samples.mean(), 
            unique1_samples.mean(), 
            unique2_samples.mean(), 
            synergy_samples.mean()
        ])
        
        # Aggressive cleanup (optional, from global config)
        if self.aggressive_cleanup and x1.is_cuda: # Check if on cuda
            # This logic for query and interval might be better in the training loop after N batches.
            # if torch.cuda.current_stream(x1.device).query() and self.memory_cleanup_interval > 0:
            # For now, just empty cache if flag is set.
            torch.cuda.empty_cache()
        
        return loss, pid_vals, P 