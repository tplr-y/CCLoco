import torch
import torch.distributed as dist
from typing import Dict
import math


def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. For efficient orthogonalization we use a Newton-Schulz iteration, which has the
    advantage that it can be stably run in bfloat16 on the GPU.

    Muon should only be used for hidden weight layers. The input embedding, final output layer,
    and any internal gains or biases should be optimized using a standard method such as AdamW.
    Hidden convolutional weights can be trained using Muon by viewing them as 2D and then
    collapsing their last 3 dimensions.

    Arguments:
        lr: The learning rate, in units of spectral norm per update.
        weight_decay: The AdamW-style weight decay.
        momentum: The momentum. A value of 0.95 here is usually fine.
    """
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
            for base_i in range(len(params))[::dist.get_world_size()]:
                if base_i + dist.get_rank() < len(params):
                    p = params[base_i + dist.get_rank()]
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])

        return loss


class SingleDeviceMuon(torch.optim.Optimizer):
    """
    Muon variant for usage in non-distributed settings.
    """
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    # continue
                    p.grad = torch.zeros_like(p)  # Force synchronization
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)


class MuonWithAuxAdam(torch.optim.Optimizer):
    """
    Distributed Muon variant that can be used for all parameters in the network, since it runs an
    internal AdamW for the parameters that are not compatible with Muon. The user must manually
    specify which parameters shall be optimized with Muon and which with Adam by passing in a
    list of param_groups with the `use_muon` flag set.

    The point of this class is to allow the user to have a single optimizer in their code, rather
    than having both a Muon and an Adam which each need to be stepped.

    You can see an example usage below:

    https://github.com/KellerJordan/modded-nanogpt/blob/master/records/052525_MuonWithAuxAdamExample/b01550f9-03d8-4a9c-86fe-4ab434f1c5e0.txt#L470
    ```
    hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
    embed_params = [p for n, p in model.named_parameters() if "embed" in n]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    head_params = [model.lm_head.weight]

    from muon import MuonWithAuxAdam
    adam_groups = [dict(params=head_params, lr=0.22), dict(params=embed_params, lr=0.6), dict(params=scalar_params, lr=0.04)]
    adam_groups = [dict(**g, betas=(0.8, 0.95), eps=1e-10, use_muon=False) for g in adam_groups]
    muon_group = dict(params=hidden_matrix_params, lr=0.05, momentum=0.95, use_muon=True)
    param_groups = [*adam_groups, muon_group]
    optimizer = MuonWithAuxAdam(param_groups)
    ```
    """
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
                for base_i in range(len(params))[::dist.get_world_size()]:
                    if base_i + dist.get_rank() < len(params):
                        p = params[base_i + dist.get_rank()]
                        if p.grad is None:
                            # continue
                            p.grad = torch.zeros_like(p)  # Force synchronization
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
                    dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """
    Non-distributed variant of MuonWithAuxAdam.
    """
    def __init__(self, param_groups, track_rms=True, use_qk_clip=False, muonclip_threshold=100.0):
        # Store parameter names for layer-specific RMS tracking
        self.param_names = {}
        
        self.use_qk_clip = use_qk_clip
        self.muonclip_threshold = muonclip_threshold

        if self.use_qk_clip:
            print("SingleDeviceMuonWithAuxAdam: Using QK clip with threshold: ", self.muonclip_threshold)
        
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())

        # Initialize scaling configuration
        self.track_rms = track_rms
        
        # RMS tracking for analysis (optional)
        self.rms_history = {} if self.track_rms else None
    
    def set_model_reference(self, model):
        """Collect attention modules for QK-Clip."""        
        self.attn_modules = []
        for layer in model.model.layers:
            if hasattr(layer, 'self_attn'):
                self.attn_modules.append(layer.self_attn)

    def set_param_names(self, param_to_name: Dict[torch.nn.Parameter, str]):
        """Set parameter names for layer-specific RMS tracking."""
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
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    
                    # Track RMS if enabled
                    if self.rms_history is not None:
                        rms = (update.norm() / math.sqrt(update.numel())).item()
                        param_name = self.param_names.get(p, f'param_{id(p)}')
                        if param_name not in self.rms_history:
                            self.rms_history[param_name] = []
                        self.rms_history[param_name].append(group["lr"] * rms)

                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
        if self.use_qk_clip:
            self._apply_qk_clip()

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

    def _apply_qk_clip(self):
        """Apply QK-Clip to Q and K projection weights if needed."""
        if not self.use_qk_clip or not hasattr(self, 'attn_modules'):
            return
        
        for attn_module in self.attn_modules:
            if not hasattr(attn_module, '_qk_clip_max_logits'):
                continue
                
            max_logits = attn_module._qk_clip_max_logits
            
            # Early exit if no clipping needed
            if (max_logits <= self.muonclip_threshold).all():
                continue
            
            # Vectorized scale computation
            mask = max_logits > self.muonclip_threshold
            scales = torch.ones_like(max_logits)
            scales[mask] = torch.sqrt(self.muonclip_threshold / max_logits[mask])
            
            # Get dimensions
            num_heads = len(max_logits)
            head_dim = attn_module.q_proj.weight.shape[0] // num_heads
            
            # Reshape, scale, reshape back
            q_weight = attn_module.q_proj.weight.view(num_heads, head_dim, -1)
            k_weight = attn_module.k_proj.weight.view(num_heads, head_dim, -1)
            
            # Apply scaling using broadcasting
            q_weight.mul_(scales.view(num_heads, 1, 1))
            k_weight.mul_(scales.view(num_heads, 1, 1))
            
            # Update weights
            attn_module.q_proj.weight.data = q_weight.view(attn_module.q_proj.weight.shape)
            attn_module.k_proj.weight.data = k_weight.view(attn_module.k_proj.weight.shape)


def monkey_patch_muon_clip(model, log):
    """
    Patch a Llama model to track max attention logits for MuonClip optimizer.

    eager_attention_forward is defined in the following:

    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L171

    We leave the attention computation unchanged.

    The only modification is the if statement that captures the max logits 
    per head before softmax, and stores them in the model's _qk_clip_max_logits dictionary. 
    This is only done if the  _track_qk_clip attribute is set to True. 
    The memory overhead is negligible. For a 2B model with 24 attention layers and 
    20 attention heads per layer that is 480 float32 values in total.
    
    Args:
        model: A LlamaForCausalLM model instance
    """
    import torch
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.llama.modeling_llama import repeat_kv
    
    def muonclip_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        scaling,
        dropout=0.0,
        **kwargs,
    ):
        """Standard attention that tracks max logits for MuonClip."""
        
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)
        
        is_causal = query.shape[2] > 1 and attention_mask is None and getattr(module, "is_causal", True)
        attn_bias = torch.zeros(query.size(-2), key.size(-2), dtype=query.dtype, device=query.device)
        if is_causal:
            temp_mask = torch.ones(query.size(-2), key.size(-2), dtype=torch.bool, device=query.device).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        attn_weights += attn_bias

        # # Store max logits per head before softmax after causal mask is applied
        if hasattr(module, '_track_qk_clip'):
            # More efficient max computation without flatten
            # attn_weights shape: [batch, num_heads, seq_len, seq_len]
            batch_size, num_heads, seq_len, _ = attn_weights.shape
            # Reshape and compute max in one operation
            max_per_head = attn_weights.view(batch_size, num_heads, -1).amax(dim=(0, 2))
            # Keep running maximum across forward passes (for gradient accumulation)
            if hasattr(module, '_qk_clip_max_logits'):
                module._qk_clip_max_logits = torch.maximum(module._qk_clip_max_logits, max_per_head.detach())
            else:
                module._qk_clip_max_logits = max_per_head.detach()
        
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
        
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout, training=True)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output, attn_weights
    
    # Mark attention modules for tracking
    for layer in model.model.layers:
        layer.self_attn._track_qk_clip = True
    
    log.info('registering monkeypatched attention; current attention implementation: %s', model.config._attn_implementation)
    # Register custom attention
    ALL_ATTENTION_FUNCTIONS.register("monkeypatched_eager_with_causal_mask_and_logit_tracking", muonclip_attention_forward)
    model.config._attn_implementation = "monkeypatched_eager_with_causal_mask_and_logit_tracking"
    log.info('now using monkeypatched attention %s', model.config._attn_implementation)