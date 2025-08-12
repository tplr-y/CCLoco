"""
Adaptive Muon implementation with configurable scaling strategies.
This extends the standard Muon optimizer to support both original scaling
and Moonlight-style RMS targeting for use as inner optimizer in CCLoco.
"""

import torch
import torch.distributed as dist
import math
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

from muon import zeropower_via_newtonschulz5, adam_update


@dataclass
class MuonScalingConfig:
    """Configuration for Muon scaling strategies."""
    mode: str = 'muon'  # 'muon' or 'moonlight'
    rms_target: float = 0.2  # Target RMS for moonlight mode
    track_rms: bool = True  # Whether to track update RMS statistics


def compute_adaptive_scale(
    grad_shape: torch.Size,
    config: Optional[MuonScalingConfig] = None
) -> float:
    """
    Compute scaling factor based on gradient shape and configuration.
    
    Args:
        grad_shape: Shape of the gradient tensor
        config: Scaling configuration
        
    Returns:
        Scaling factor to apply to the orthogonalized update
    """
    if config is None:
        config = MuonScalingConfig()
        
    if len(grad_shape) < 2:
        return 1.0
        
    m, n = grad_shape[-2], grad_shape[-1]
    
    mode = config.mode
    if mode == 'muon':
        # Original Muon scaling: sqrt(fan_out/fan_in)
        return max(1, m / n)**0.5
        
    if mode == 'moonlight':
        # Moonlight scaling: fixed coefficient * sqrt(max_dim)
        return config.rms_target * math.sqrt(max(m, n))
        
    else:
        raise ValueError(f"Unknown scaling mode: {mode}")


def muon_update_adaptive(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    beta: float = 0.95,
    ns_steps: int = 5,
    nesterov: bool = True,
    scaling_config: Optional[MuonScalingConfig] = None
) -> torch.Tensor:
    """
    Muon update with adaptive scaling support.
    
    Args:
        grad: Gradient tensor
        momentum: Momentum buffer
        beta: Momentum coefficient
        ns_steps: Number of Newton-Schulz iterations
        nesterov: Whether to use Nesterov momentum
        scaling_config: Configuration for scaling strategy
        
    Returns:
        Scaled orthogonalized update
    """
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    
    if update.ndim == 4:  # Conv filters
        update = update.view(len(update), -1)
        
    # Orthogonalization via Newton-Schulz
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    
    # Apply adaptive scaling
    scale = compute_adaptive_scale(grad.shape, scaling_config)
    update *= scale
    
    return update


class SingleDeviceMuonWithAuxAdamAdaptive(torch.optim.Optimizer):
    """
    Enhanced version of SingleDeviceMuonWithAuxAdam with adaptive scaling support.
    Maintains backward compatibility while adding new scaling options.
    """
    
    def __init__(self, param_groups, scaling_config: Optional[MuonScalingConfig] = None):
        # Store parameter names for layer-specific scaling
        self.param_names = {}
        
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                # defaults
                group["lr"] = group.get("lr")#, 0.02)
                group["momentum"] = group.get("momentum")#, 0.95)
                group["weight_decay"] = group.get("weight_decay")#, 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                # defaults for Adam
                group["lr"] = group.get("lr")#, 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay")#, 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
                
        super().__init__(param_groups, dict())
        
        # Initialize scaling configuration
        self.scaling_config = scaling_config or MuonScalingConfig()
        
        # RMS tracking for analysis (optional)
        self.rms_history = {} if self.scaling_config.track_rms else None
        
    def set_param_names(self, param_to_name: Dict[torch.nn.Parameter, str]):
        """Set parameter names for layer-specific scaling."""
        self.param_names = param_to_name
        
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)  # Force synchronization
                        
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                        
                    # Adaptive Muon update
                    update = muon_update_adaptive(
                        p.grad,
                        state["momentum_buffer"],
                        beta=group["momentum"],
                        scaling_config=self.scaling_config
                    )
                    
                    # Track RMS if enabled
                    if self.rms_history is not None:
                        rms = (update.norm() / math.sqrt(update.numel())).item()
                        param_name = self.param_names.get(p, f'param_{id(p)}')
                        if param_name not in self.rms_history:
                            self.rms_history[param_name] = []
                        self.rms_history[param_name].append(group["lr"] * rms)
                    
                    # Apply update
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                # Standard Adam path (unchanged)
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)  # Force synchronization
                        
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                        
                    state["step"] += 1
                    update = adam_update(
                        p.grad, state["exp_avg"], state["exp_avg_sq"],
                        state["step"], group["betas"], group["eps"]
                    )
                    
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss
    
    def get_rms_stats(self) -> Dict[str, Dict[str, float]]:
        """Get RMS statistics by layer type."""
        if self.rms_history is None:
            return {}
            
        import numpy as np
        stats = {}
        
        # Group by layer type
        layer_types = set()
        for param_name in self.rms_history:
            for pattern in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 
                          'up_proj', 'gate_proj', 'down_proj']:
                if pattern in param_name:
                    layer_types.add(pattern)
                    break
                    
        for layer_type in layer_types:
            values = []
            for param_name, rms_list in self.rms_history.items():
                if layer_type in param_name:
                    values.extend(rms_list[-100:])  # Last 100 values
                    
            if values:
                stats[layer_type] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                }
                
        return stats