import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import itertools
from typing import List, Callable
from contextlib import nullcontext

# Import AMP functionality
try:
    from torch.amp import autocast, GradScaler
    
    class DummyAMPModule:
        def __init__(self):
            self.autocast = autocast
            self.GradScaler = GradScaler
    
    amp = DummyAMPModule()
except ImportError:
    # Fallback for older PyTorch versions
    from contextlib import nullcontext
    
    class DummyAMPModule:
        def __init__(self):
            self.autocast = nullcontext
            self.GradScaler = None
    
    amp = DummyAMPModule()

from .sinkhorn import sinkhorn_probs

# Global configurations (avoid circular imports by defining locally)
global_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = False  # Default value
PRECISION = torch.float16  # Default precision for AMP
CHUNK_SIZE = 128  # Default chunk size
AGGRESSIVE_CLEANUP = False  # Whether to aggressively clean memory
MEMORY_CLEANUP_INTERVAL = 10  # Clean memory every N chunks

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
    return nn.Sequential(*modules) #.to(global_device) -> model that uses it will be moved to device #TODO check if this is correct

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

class PretrainedDiscrim(nn.Module):
    """Discriminator using pretrained frozen encoder with additional softmax layers."""
    
    def __init__(
        self,
        pretrained_encoder: nn.Module,
        num_labels: int,
        hidden_dim: int = 64,
        layers: int = 2,
        activation: str = 'relu'
    ):
        super().__init__()
        
        # Store the pretrained encoder and freeze it
        self.pretrained_encoder = pretrained_encoder
        for param in self.pretrained_encoder.parameters():
            param.requires_grad = False
        self.pretrained_encoder.eval()
        
        # Get the output dimension of the pretrained encoder
        # We'll infer this during the first forward pass
        self.encoder_output_dim = None
        self.classifier = None
        
        # Store parameters for lazy initialization
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.activation = activation
    
    def _initialize_classifier(self, encoder_output_dim: int):
        """Initialize the classifier MLP once we know the encoder output dimension."""
        if self.layers == 1:
            # Single layer: direct classification
            self.classifier = nn.Linear(encoder_output_dim, self.num_labels)
        else:
            # Two-layer MLP as requested
            act_map = {'relu': nn.ReLU, 'tanh': nn.Tanh}
            act_layer = act_map.get(self.activation, nn.ReLU)
            
            modules = [ #TODO this is not written generally, it is hardcoded for 2 layers. Use MLP for it, but be careful with dim matching.
                nn.Linear(encoder_output_dim, self.hidden_dim),
                act_layer(),
                nn.Linear(self.hidden_dim, self.num_labels)
            ]
            self.classifier = nn.Sequential(*modules)
        
        # Move to the same device as the encoder
        device = next(self.pretrained_encoder.parameters()).device
        self.classifier = self.classifier.to(device)
    
    def trainable_parameters(self):
        """Return only the parameters of the classifier (not the frozen encoder)."""
        if self.classifier is not None:
            return self.classifier.parameters()
        else:
            # Return empty generator if classifier not initialized yet
            return iter([])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass through frozen pretrained encoder
        with torch.no_grad():
            encoded = self.pretrained_encoder(x)
        
        # Initialize classifier if not done yet
        if self.classifier is None:
            self.encoder_output_dim = encoded.size(-1)
            self._initialize_classifier(self.encoder_output_dim)
        
        # Pass through trainable classifier
        logits = self.classifier(encoded)
        return logits

class PretrainedJointDiscrim(nn.Module):
    """Joint discriminator that concatenates outputs from two pretrained encoders."""
    
    def __init__(
        self,
        pretrained_encoder1: nn.Module,
        pretrained_encoder2: nn.Module,
        num_labels: int,
        hidden_dim: int = 64,
        layers: int = 2,
        activation: str = 'relu'
    ):
        super().__init__()
        
        # Store the pretrained encoders and freeze them
        self.pretrained_encoder1 = pretrained_encoder1
        self.pretrained_encoder2 = pretrained_encoder2
        
        for encoder in [self.pretrained_encoder1, self.pretrained_encoder2]:
            for param in encoder.parameters():
                param.requires_grad = False
            encoder.eval()
        
        # Get the combined output dimension of the pretrained encoders
        # We'll infer this during the first forward pass
        self.combined_output_dim = None
        self.classifier = None
        
        # Store parameters for lazy initialization
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.activation = activation
    
    def _initialize_classifier(self, combined_output_dim: int):
        """Initialize the classifier MLP once we know the combined encoder output dimension."""
        if self.layers == 1:
            # Single layer: direct classification
            self.classifier = nn.Linear(combined_output_dim, self.num_labels)
        else:
            # Two-layer MLP as requested #TODO this is not written generally, it is hardcoded for 2 layers. Use MLP for it, but be careful with dim matching.
            act_map = {'relu': nn.ReLU, 'tanh': nn.Tanh}
            act_layer = act_map.get(self.activation, nn.ReLU)
            
            modules = [
                nn.Linear(combined_output_dim, self.hidden_dim),
                act_layer(),
                nn.Linear(self.hidden_dim, self.num_labels)
            ]
            self.classifier = nn.Sequential(*modules)
        
        # Move to the same device as the encoders
        device = next(self.pretrained_encoder1.parameters()).device
        self.classifier = self.classifier.to(device)
    
    def trainable_parameters(self):
        """Return only the parameters of the classifier (not the frozen encoders)."""
        if self.classifier is not None:
            return self.classifier.parameters()
        else:
            # Return empty generator if classifier not initialized yet
            return iter([])
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # Pass through frozen pretrained encoders
        with torch.no_grad():
            encoded1 = self.pretrained_encoder1(x1)
            encoded2 = self.pretrained_encoder2(x2)
        
        # Concatenate the encoded representations
        combined = torch.cat([encoded1, encoded2], dim=-1)
        
        # Initialize classifier if not done yet
        if self.classifier is None:
            self.combined_output_dim = combined.size(-1)
            self._initialize_classifier(self.combined_output_dim)
        
        # Pass through trainable classifier
        logits = self.classifier(combined)
        return logits

class CEAlignment(nn.Module): #TODO give another name to this class
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
        self.mlp1 = mlp(x1_dim, hidden_dim, embed_dim * num_labels, layers, activation) #TODO think carefully about the scaling as the num clusters increases... do we have enough data?
        self.mlp2 = mlp(x2_dim, hidden_dim, embed_dim * num_labels, layers, activation)
        #TODO find out if the embed dim times num labels is correct. On a general level, code is fine, but approach may not be.
        #TODO find out where the embed dim comes from. And generally the params of these networks, I think it is not CLI, so it must be some arbitrary default in main.py
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        p_y_x1: torch.Tensor,
        p_y_x2: torch.Tensor
    ) -> torch.Tensor:
        batch_size = x1.size(0)
        if x2.size(0) != batch_size:
            raise ValueError(f"Batch size mismatch: x1 has batch size {batch_size} but x2 has batch size {x2.size(0)}. Inputs must have matching batch dimensions.")
        
        q1 = self.mlp1(x1).unflatten(1, (self.num_labels, -1)) 
        q2 = self.mlp2(x2).unflatten(1, (self.num_labels, -1))

        q1 = (q1 - q1.mean(dim=2, keepdim=True)) / torch.sqrt(q1.var(dim=2, keepdim=True) + 1e-8)
        q2 = (q2 - q2.mean(dim=2, keepdim=True)) / torch.sqrt(q2.var(dim=2, keepdim=True) + 1e-8)
        #TODO study how einsum works in depth
        aff = torch.einsum('bce, dce -> bdc', q1, q2) / math.sqrt(q1.size(-1))
        aff_max = aff.reshape(-1, aff.size(-1)).max(dim=0)[0]
        aff_centered = aff - aff_max.view(1, 1, -1) #TODO check if they also do this in the paper, if not investigate more
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
        return P

class CEAlignmentInformation(nn.Module): #TODO give another name to this class
    def __init__(self, x1_dim, x2_dim, hidden_dim, embed_dim, num_labels,
                 layers, activation, discrim_1, discrim_2, discrim_12, p_y):
        
        super().__init__()
        self.num_labels = num_labels
        self.align = CEAlignment(x1_dim, x2_dim, hidden_dim, embed_dim, num_labels, layers, activation)
        
        # Store discriminators
        self.discrim_1 = discrim_1
        self.discrim_2 = discrim_2
        self.discrim_12 = discrim_12
        
        # Freeze pre-trained discriminators
        for D in (self.discrim_1, self.discrim_2, self.discrim_12):
            D.eval()
            for p in D.parameters():
                p.requires_grad = False
                
        # marginal p(y)
        self.register_buffer('p_y', p_y) #TODO where do we calculate Py? need to plot it, also.
        #TODO find out if mixed precision can be savely removed
        # For mixed precision - import USE_AMP from utils
        try:
            from .utils import USE_AMP, CHUNK_SIZE, AGGRESSIVE_CLEANUP, MEMORY_CLEANUP_INTERVAL
            self.use_amp = USE_AMP
            self.chunk_size = CHUNK_SIZE
            self.aggressive_cleanup = AGGRESSIVE_CLEANUP
            self.memory_cleanup_interval = MEMORY_CLEANUP_INTERVAL
        except ImportError:
            # Fallback to default values
            self.use_amp = False
            self.chunk_size = 128
            self.aggressive_cleanup = False
            self.memory_cleanup_interval = 10
        
        # Create GradScaler for mixed precision training
        if self.use_amp:
            try:
                from torch.cuda.amp import GradScaler
                self.scaler = GradScaler()
            except ImportError:
                self.use_amp = False
                self.scaler = None
        else:
            self.scaler = None

    def forward(self, x1, x2, y):
        """
        Compute PID components between domain representations.
        
        Args:
            x1: First domain features, shape [batch_size, x1_dim]
            x2: Second domain features, shape [batch_size, x2_dim]
            y: Labels, shape [batch_size]
            
        Returns:
            Tuple of (loss, pid_vals, P) where:
            - loss: Scalar tensor for optimization
            - pid_vals: Tensor of shape [4] containing [redundancy, unique1, unique2, synergy]
            - P: Coupling matrix
        """
        batch_size = x1.size(0)
        
        # Forward pass with mixed precision if enabled
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Ensure nullcontext is available in this scope
        try:
            from torch.amp import autocast
        except ImportError:
            # Fallback for older PyTorch versions
            autocast = nullcontext  # Use the nullcontext imported at module level
        
        with autocast(device_type) if self.use_amp else nullcontext():
            # Get label conditional probabilities using discriminators with proper softmax
            p_y_x1 = F.softmax(self.discrim_1(x1), dim=1)  # [batch, num_labels]
            p_y_x2 = F.softmax(self.discrim_2(x2), dim=1)  # [batch, num_labels]
            
            # Handle joint discriminator based on its type
            if isinstance(self.discrim_12, PretrainedJointDiscrim):
                # PretrainedJointDiscrim expects two separate arguments
                p_y_x1x2 = F.softmax(self.discrim_12(x1, x2), dim=1)  # [batch, num_labels]
            else:
                # Regular Discrim expects concatenated input
                p_y_x1x2 = F.softmax(self.discrim_12(torch.cat([x1, x2], dim=-1)), dim=1)  # [batch, num_labels]
        
        #TODO understand the math here and below (didn't do yet), if complete and correct
        # Calculate unimodal mutual information terms with explicit dtype for numerical stability
        mi_x1_y = torch.sum(p_y_x1 * torch.log(p_y_x1 / self.p_y.unsqueeze(0) + 1e-8), dim=1, dtype=torch.float32)
        mi_x2_y = torch.sum(p_y_x2 * torch.log(p_y_x2 / self.p_y.unsqueeze(0) + 1e-8), dim=1, dtype=torch.float32)
        
        # Get coupling matrix from align
        P = self.align(x1, x2, p_y_x1, p_y_x2)  # [batch, batch, num_labels]
        
        # 1) Normalize along the label axis to get q̃(y|x1,x2)
        P_cond = P / (P.sum(dim=-1, keepdim=True) + 1e-8)  # [batch, batch, num_labels]
        
        # 2) Compute the joint mutual information properly using p(y) as denominator
        # Expand p_y for broadcasting
        p_y_expanded = self.p_y.view(1, 1, -1)  # [1, 1, num_labels]
        
        # Calculate proper log ratio for joint MI
        log_ratio = torch.log(P_cond + 1e-8) - torch.log(p_y_expanded + 1e-8)  # [batch, batch, num_labels]
        
        # 3) Compute joint MI by summing over all dimensions, weighted by joint coupling P
        mi_x1x2_y = (P * log_ratio).sum(dim=[1, 2])  # [batch]
        
        # For comparison - calculate joint MI from discriminator (not used in updated PID calculation)
        mi_discrim_x1x2_y = torch.sum(p_y_x1x2 * torch.log(p_y_x1x2 / self.p_y.unsqueeze(0) + 1e-8), 
                                     dim=1, dtype=torch.float32)
        
        # Calculate PID components using the Möbius relations
        # Redundancy = I(X1;Y) + I(X2;Y) - I(X1,X2;Y)
        redundancy = torch.clamp(mi_x1_y + mi_x2_y - mi_x1x2_y, min=0)
        # Unique1 = I(X1;Y) - Redundancy
        unique1 = torch.clamp(mi_x1_y - redundancy, min=0)
        # Unique2 = I(X2;Y) - Redundancy
        unique2 = torch.clamp(mi_x2_y - redundancy, min=0)
        
        # 🔬 CRITICAL MATHEMATICAL CORRECTION 1: Updated Synergy Calculation
        # Compute the *data* joint‐MI via your joint discriminator
        mi_p_y_x1x2 = mi_discrim_x1x2_y.mean()
        mi_q_y_x1x2 = mi_x1x2_y.mean()

        # Synergy = I_p(X1,X2;Y) − I_q(X1,X2;Y)
        synergy = torch.clamp(mi_p_y_x1x2 - mi_q_y_x1x2, min=0)
        
        
        loss = mi_q_y_x1x2  # Optimize the coupling-based joint MI
        
        # Final cleanup
        if self.aggressive_cleanup and torch.cuda.is_available():
            if torch.cuda.current_stream().query() and self.memory_cleanup_interval > 0:
                torch.cuda.empty_cache()
        
        # Return loss, PID components, and coupling matrix
        pid_vals = torch.stack([redundancy.mean(), unique1.mean(), unique2.mean(), synergy])
        return loss, pid_vals, P 