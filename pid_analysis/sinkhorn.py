import torch
import torch.utils.checkpoint
import gc
from typing import Optional

# Global configurations (avoid circular imports by defining locally)
USE_AMP = False  # Default value
PRECISION = torch.float16  # Default precision for AMP
CHUNK_SIZE = 128  # Default chunk size
MEMORY_CLEANUP_INTERVAL = 10  # Clean memory every N chunks
AGGRESSIVE_CLEANUP = False  # Whether to aggressively clean memory

def sinkhorn_probs(
    matrix: torch.Tensor,
    x1_probs: torch.Tensor,
    x2_probs: torch.Tensor,
    tol: float = 1e-8,
    max_iter: int = 500,
    chunk_size: Optional[int] = None, # Parameter chunk_size takes precedence if provided
) -> torch.Tensor:
    """
    Apply the Sinkhorn-Knopp algorithm to project a matrix onto a transport polytope.
    Uses out-of-place operations to preserve gradients and chunked checkpointing.
    
    Args:
        matrix: Input matrix to project, shape [m, n]
        x1_probs: Target row sum distribution, shape [m]
        x2_probs: Target column sum distribution, shape [n]
        tol: Convergence tolerance for row/column sums
        max_iter: Maximum number of iterations
        chunk_size: Size of chunks for checkpointing (default: sqrt(max_iter) or CHUNK_SIZE if param is None)
        
    Returns:
        Projected matrix that approximately satisfies the marginal constraints
    """
    # Use local AMP and PRECISION settings
    if USE_AMP:
        dtype = PRECISION if matrix.is_cuda else torch.float32
    else:
        dtype = torch.float32
    
    matrix = matrix.to(dtype)
    x1_probs = x1_probs.to(dtype)
    x2_probs = x2_probs.to(dtype)
    
    if x2_probs.size(0) != matrix.size(1):
        if x2_probs.size(0) == 1:
            x2_probs = x2_probs.expand(matrix.size(1))
        else:
            if matrix.size(1) > x2_probs.size(0):
                padding = torch.zeros(matrix.size(1) - x2_probs.size(0), device=x2_probs.device, dtype=x2_probs.dtype)
                x2_probs_padded = torch.cat([x2_probs, padding], dim=0)
                x2_probs = x2_probs_padded / (x2_probs_padded.sum(dtype=dtype) + 1e-8)
            else:
                if matrix.size(1) < x2_probs.size(0) - 128:
                    raise ValueError(f"Dimension mismatch in Sinkhorn projection: matrix columns ({matrix.size(1)}) << x2_probs size ({x2_probs.size(0)}). This is likely an error.")
                x2_probs = x2_probs[:matrix.size(1)]
                x2_probs = x2_probs / (x2_probs.sum(dtype=dtype) + 1e-8)
    
    if x1_probs.size(0) != matrix.size(0):
        if x1_probs.size(0) == 1:
            x1_probs = x1_probs.expand(matrix.size(0))
        else:
            if matrix.size(0) > x1_probs.size(0):
                padding = torch.zeros(matrix.size(0) - x1_probs.size(0), device=x1_probs.device, dtype=x1_probs.dtype)
                x1_probs_padded = torch.cat([x1_probs, padding], dim=0)
                x1_probs = x1_probs_padded / (x1_probs_padded.sum(dtype=dtype) + 1e-8)
            else:
                if matrix.size(0) < x1_probs.size(0) - 128: # Reasonable chunk size threshold
                    raise ValueError(f"Dimension mismatch in Sinkhorn projection: matrix rows ({matrix.size(0)}) << x1_probs size ({x1_probs.size(0)}). This is likely an error.")
                x1_probs = x1_probs[:matrix.size(0)]
                x1_probs = x1_probs / (x1_probs.sum(dtype=dtype) + 1e-8)
    
    def sinkhorn_chunk(mat: torch.Tensor, n_steps: int) -> torch.Tensor:
        for _ in range(n_steps):
            col_sums = mat.sum(dim=0, keepdim=True, dtype=dtype)
            mat = mat / (col_sums + 1e-8) * x2_probs.unsqueeze(0)
            row_sums = mat.sum(dim=1, keepdim=True, dtype=dtype)
            mat = mat / (row_sums + 1e-8) * x1_probs.unsqueeze(1)
        return mat
    
    # Determine effective chunk_size: parameter > original_default > global_fallback
    if chunk_size is not None:
        effective_chunk_size = chunk_size
    else: # chunk_size parameter is None, use original default behavior
        effective_chunk_size = int(max_iter ** 0.5)
        # As a secondary fallback, if one wanted to use CHUNK_SIZE when param is None and sqrt(max_iter) is not desired:
        # if CHUNK_SIZE is not None:
        #     effective_chunk_size = CHUNK_SIZE

    done = 0
    while done < max_iter:
        steps = min(effective_chunk_size, max_iter - done)
        matrix = torch.utils.checkpoint.checkpoint(
            lambda m, s=steps: sinkhorn_chunk(m, s),
            matrix,
            use_reentrant=False
        )
        done += steps
        
        if (torch.allclose(matrix.sum(dim=1, dtype=dtype), x1_probs, atol=tol) and
            torch.allclose(matrix.sum(dim=0, dtype=dtype), x2_probs, atol=tol)):
            break
        
        # Use local MEMORY_CLEANUP_INTERVAL and AGGRESSIVE_CLEANUP
        if done % (effective_chunk_size * MEMORY_CLEANUP_INTERVAL) == 0 and AGGRESSIVE_CLEANUP:
            torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
    
    return matrix.to(torch.float32) 