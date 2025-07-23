"""
CCLoco: Chunk-Compressed Low-Communication Optimization.

This module provides an implementation of the CCLoco algorithm, a memory-efficient,
low-bandwidth optimizer for distributed training.

The core method is chunked Top-K sparsification with error feedback. Before
communication, each parameter's gradient is chunked, and only the most significant
values from each chunk are selected for transmission. The remaining values (the
compression error) are accumulated locally and reapplied in the next step.

This implementation also includes an optional DCT-based compression variant to
allow for result reproduction with the original DeMo optimizer.

Code adapted from: https://github.com/bloc97/DeMo
"""

# Standard library
import math
from typing import Optional, Callable, List, Tuple, Union, TypeAlias, Iterable, Any

# Third party
import torch
import torch.fft
import torch.distributed as dist
from einops import rearrange

# ---------- Type Aliases ---------- #
ParamsT: TypeAlias = Union[Iterable[torch.Tensor], Iterable[dict[str, Any]]]


def _dct_fft_impl(v):
    return torch.view_as_real(torch.fft.fft(v, dim=1))


def _idct_irfft_impl(V):
    return torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)


def _dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = _dct_fft_impl(v)

    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * math.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == "ortho":
        V[:, 0] /= math.sqrt(N) * 2
        V[:, 1:] /= math.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def _idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct(dct(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == "ortho":
        X_v[:, 0] *= math.sqrt(N) * 2
        X_v[:, 1:] *= math.sqrt(N / 2) * 2

    k = (
        torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :]
        * math.pi
        / (2 * N)
    )
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = _idct_irfft_impl(V)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, : N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, : N // 2]

    return x.view(*x_shape)


def _get_prime_divisors(n):
    divisors = []
    while n % 2 == 0:
        divisors.append(2)
        n //= 2
    while n % 3 == 0:
        divisors.append(3)
        n //= 3
    i = 5
    while i * i <= n:
        for k in (i, i + 2):
            while n % k == 0:
                divisors.append(k)
                n //= k
        i += 6
    if n > 1:
        divisors.append(n)
    return divisors


def _get_divisors(n):
    divisors = []
    if n == 1:
        divisors.append(1)
    elif n > 1:
        prime_factors = _get_prime_divisors(n)
        divisors = [1]
        last_prime = 0
        factor = 0
        slice_len = 0
        # Find all the products that are divisors of n
        for prime in prime_factors:
            if last_prime != prime:
                slice_len = len(divisors)
                factor = prime
            else:
                factor *= prime
            for i in range(slice_len):
                divisors.append(divisors[i] * factor)
            last_prime = prime
        divisors.sort()
    return divisors


def _get_smaller_split(n, close_to):
    all_divisors = _get_divisors(n)
    for ix, val in enumerate(all_divisors):
        if val == close_to:
            return val
        if val > close_to:
            if ix == 0:
                return val
            return all_divisors[ix - 1]
    return n


class ChunkingTransform:
    """Handles tensor chunking, with an optional DCT transformation."""

    def __init__(
        self, param_groups: ParamsT, chunk_size: int, use_dct: bool, norm: str = "ortho"
    ):
        self.target_chunk = chunk_size
        self.use_dct = use_dct
        self.shape_dict = {}
        self.f_dict, self.b_dict = (
            {},
            {},
        )  #  Forward (DCT) and backward (IDCT) transform matrices
        self._initialize_transforms(param_groups, norm)

    def _initialize_transforms(self, param_groups: ParamsT, norm: str):
        """Precomputes chunk shapes and, if enabled, DCT bases."""
        for group in param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                for s in p.shape:
                    if s not in self.shape_dict:
                        # Chunking related
                        sc = _get_smaller_split(s, self.target_chunk)
                        self.shape_dict[s] = sc
                        # DCT related
                        if self.use_dct and sc not in self.f_dict:
                            I = torch.eye(sc)
                            self.f_dict[sc] = (
                                _dct(I, norm=norm).to(p.dtype).to(p.device)
                            )
                            self.b_dict[sc] = (
                                _idct(I, norm=norm).to(p.dtype).to(p.device)
                            )

    def einsum_2d(self, x, b, d=None):
        if d is None:
            return torch.einsum("...ij, jb -> ...ib", x, b)
        else:
            # Note: b-c axis output is transposed to chunk DCT in 2D
            return torch.einsum("...ijkl, kb, ld -> ...ijbd", x, b, d)

    def einsum_2d_t(self, x, b, d=None):
        if d is None:
            return torch.einsum("...ij, jb -> ...ib", x, b)
        else:
            # Note: b-c axis output is transposed to chunk DCT in 2D
            return torch.einsum("...ijbd, bk, dl -> ...ijkl", x, b, d)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Chunks a tensor and optionally applies DCT."""
        if len(x.shape) > 1:  # 2D weights
            n1 = self.shape_dict[x.shape[0]]
            n2 = self.shape_dict[x.shape[1]]
            x = rearrange(x, "(y h) (x w) -> y x h w", h=n1, w=n2)
            if not self.use_dct:
                return x

            n1w = self.f_dict[n1].to(x.device)
            n2w = self.f_dict[n2].to(x.device)
            self.f_dict[n1] = n1w
            self.f_dict[n2] = n2w
            x = self.einsum_2d(x, n1w, n2w)

        else:  # 1D weights
            n1 = self.shape_dict[x.shape[0]]
            x = rearrange(x, "(x w) -> x w", w=n1)
            if not self.use_dct:
                return x

            n1w = self.f_dict[n1].to(x.device)
            self.f_dict[n1] = n1w
            x = self.einsum_2d(x, n1w)

        return x

    @torch.no_grad()
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """De-chunks a tensor and optionally applies inverse DCT."""
        if len(x.shape) > 2:  # 2D weights
            if self.use_dct:
                n1 = x.shape[2]
                n2 = x.shape[3]
                n1w = self.b_dict[n1].to(x.device)
                n2w = self.b_dict[n2].to(x.device)
                self.b_dict[n1] = n1w
                self.b_dict[n2] = n2w
                x = self.einsum_2d_t(x, n1w, n2w)
            x = rearrange(x, "y x h w -> (y h) (x w)")
        else:  # 1D weights
            if self.use_dct:
                n1 = x.shape[1]
                n1w = self.b_dict[n1].to(x.device)
                self.b_dict[n1] = n1w
                x = self.einsum_2d_t(x, n1w)

            x = rearrange(x, "x w -> (x w)")

        return x


class TopKCompressor:
    """Handles Top-K sparsification and optional statistical quantization."""

    def __init__(self, use_quantization: bool, n_bins: int, range_in_sigmas: float):
        self.use_quantization = use_quantization
        if use_quantization:
            self.n_bins = n_bins
            self.range_in_sigmas = range_in_sigmas

    def _clamp_topk(self, x, topk):
        if topk > x.shape[-1]:
            topk = x.shape[-1]
        if topk < 1:
            topk = 1
        return topk

    @torch.no_grad()
    def compress(self, x: torch.Tensor, k: int):
        """Selects Top-K values from each chunk and optionally quantizes them."""
        x_shape = x.shape
        if len(x.shape) > 2:  # 2D weights
            x_flat_chunks = rearrange(x, "y x h w -> y x (h w)")
        else:
            x_flat_chunks = rearrange(x, "x w -> x w")

        # Limit topk to max size
        k = self._clamp_topk(x_flat_chunks, k)

        _, idx = torch.topk(
            x_flat_chunks.abs(), k=k, dim=-1, largest=True, sorted=False
        )
        val = torch.gather(x_flat_chunks, dim=-1, index=idx)

        quant_params = None
        if self.use_quantization:
            quantized_val, quant_params = self._quantize(val)
            val = quantized_val

        return idx.to(torch.int64), val, x_shape, quant_params

    @torch.no_grad()
    def decompress(
        self,
        idx: torch.Tensor,
        val: torch.Tensor,
        x_shape: Tuple,
        ref_param: torch.Tensor,
        quant_params: Optional[Tuple],
    ) -> torch.Tensor:
        """Reconstructs a sparse tensor from indices and values."""
        if quant_params is not None:
            val = self._dequantize(val, quant_params)

        x = torch.zeros(x_shape, device=ref_param.device, dtype=ref_param.dtype)

        if len(x_shape) > 2:  # 2D weights
            x_flat = rearrange(x, "y x h w -> y x (h w)")
        else:
            x_flat = x

        x_flat.scatter_reduce_(
            dim=-1,
            index=idx.to(torch.int64),
            src=val.to(ref_param.dtype),
            reduce="mean",
            include_self=False,
        )

        if len(x_shape) > 2:  # 2D weights
            x = rearrange(x_flat, "y x (h w) -> y x h w", h=x_shape[2])
        else:
            x = x_flat

        return x

    @torch.no_grad()
    def batch_decompress(
        self, idx_list: list, val_list: list, x_shape: Tuple, ref_param: torch.Tensor
    ) -> torch.Tensor:
        """Reconstructs a sparse tensor by aggregating from a list of indices and values."""
        idx_all = torch.cat([i.to(ref_param.device) for i in idx_list], dim=-1)
        val_all = torch.cat(
            [v.to(ref_param.device, ref_param.dtype) for v in val_list], dim=-1
        )

        x = torch.zeros(x_shape, device=ref_param.device, dtype=ref_param.dtype)

        if len(x_shape) > 2:  # 2D weights
            x_flat = rearrange(x, "y x h w -> y x (h w)")
        else:
            x_flat = x

        x_flat.scatter_reduce_(
            dim=-1,
            index=idx_all.to(torch.int64),
            src=val_all,
            reduce="mean",
            include_self=False,
        )

        if len(x_shape) > 2:  # 2D weights
            x = rearrange(x_flat, "y x (h w) -> y x h w", h=x_shape[2])
        else:
            x = x_flat

        return x

    def _quantize(self, val: torch.Tensor):
        """Performs statistical 8-bit quantization."""
        offset = self.n_bins // 2
        shift = val.mean()
        centered_val = val - shift

        if centered_val.numel() <= 1:
            std_unbiased = torch.tensor(0.0, device=val.device, dtype=val.dtype)
        else:
            std_unbiased = centered_val.norm() / math.sqrt(centered_val.numel() - 1)

        scale = self.range_in_sigmas * std_unbiased / self.n_bins
        if scale == 0 or torch.isnan(scale) or torch.isinf(scale):
            scale = torch.tensor(1.0, dtype=centered_val.dtype, device=val.device)

        quantized = (
            (centered_val.float() / scale + offset)
            .round()
            .clamp(0, self.n_bins - 1)
            .to(torch.uint8)
        )

        lookup = torch.zeros(self.n_bins, dtype=torch.float32, device=val.device)
        sums = torch.zeros_like(lookup).scatter_add_(
            0, quantized.long().flatten(), centered_val.float().flatten()
        )
        counts = torch.zeros_like(lookup).scatter_add_(
            0,
            quantized.long().flatten(),
            torch.ones_like(centered_val.float().flatten()),
        )
        lookup = torch.where(counts > 0, sums / counts, 0.0)

        params_tuple = (shift, float(scale), offset, lookup, val.dtype)
        return quantized, params_tuple

    def _dequantize(self, val: torch.Tensor, quant_params: Tuple):
        """Dequantizes values using a lookup table and shift."""
        if quant_params is None:
            return val
        shift, _, _, lookup, orig_dtype = quant_params
        dequantized = lookup.to(val.device)[val.long()] + shift
        return dequantized.to(orig_dtype)


class CCLoco(torch.optim.SGD):
    """Implements the Chunk-Compressed Low-Communication (CCLoco) optimizer."""

    def __init__(
        self,
        params: ParamsT,
        lr: float,
        error_decay: float = 0.999,
        top_k: int = 32,
        chunk_size: int = 64,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        use_dct: bool = False,
        use_sign: bool = False,
        use_quantization: bool = False,
        quantization_bins: int = 256,
        quantization_range: int = 6,
        process_group: Optional[dist.ProcessGroup] = None,
        **kwargs,
    ):
        super().__init__(params, lr=lr, momentum=momentum, weight_decay=0.0, **kwargs)

        self.error_decay = error_decay
        self.top_k = top_k
        self.decoupled_weight_decay = weight_decay
        self.use_dct = use_dct
        self.use_sign = use_sign
        self.process_group = process_group

        self.transform = ChunkingTransform(self.param_groups, chunk_size, use_dct)
        self.compressor = TopKCompressor(
            use_quantization, quantization_bins, quantization_range
        )

        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    self.state[p]["error_buffer"] = torch.zeros_like(p)

    def _all_gather_tensor(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """Gathers a tensor from all workers in the process group."""
        ws = dist.get_world_size(self.process_group)
        tensor_list = [torch.zeros_like(tensor) for _ in range(ws)]
        dist.all_gather(tensor_list, tensor, group=self.process_group)
        return tensor_list

    def _all_gather_quant_params(self, quant_params: Tuple) -> List[Tuple]:
        """Gathers and reconstructs quantization parameters from all workers."""
        (shift, scale, offset, lookup, dtype) = quant_params
        comm_tensor = torch.cat([shift.view(1), lookup.to(shift.device)])
        comm_list = self._all_gather_tensor(comm_tensor)
        return [(t[0].unsqueeze(0), scale, offset, t[1:], dtype) for t in comm_list]

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Performs a single optimization step."""
        if closure:
            closure()

        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                # 1. Apply decoupled weight decay.
                if self.decoupled_weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * self.decoupled_weight_decay)

                state = self.state[p]
                error_buffer = state["error_buffer"]

                # 2. Update the error buffer: e_t = decay * e_{t-1} + lr * g_t.
                if self.error_decay != 1.0:
                    error_buffer.mul_(self.error_decay)
                error_buffer.add_(p.grad, alpha=lr)

                # 3. Compress the error buffer to get the values to transmit.
                tensor_to_compress = self.transform.encode(error_buffer)
                indices, values, shape, local_quant_params = self.compressor.compress(
                    tensor_to_compress, self.top_k
                )

                # 4. Reconstruct the local gradient to compute the error feedback.
                local_reconstruction = self.compressor.decompress(
                    indices, values, shape, p, local_quant_params
                )
                transmitted_gradient = self.transform.decode(local_reconstruction)
                error_buffer.sub_(transmitted_gradient)

                # 5. Communicate the sparse gradients (and quantization params) across all workers.
                gathered_quant_params = (
                    self._all_gather_quant_params(local_quant_params)
                    if self.compressor.use_quantization
                    and local_quant_params is not None
                    else None
                )
                gathered_indices = self._all_gather_tensor(indices)
                gathered_values = self._all_gather_tensor(values)

                # 6. Decompress and aggregate gradients from all workers.
                if self.compressor.use_quantization:
                    gathered_values = [
                        self.compressor._dequantize(v, qp)
                        for v, qp in zip(gathered_values, gathered_quant_params)
                    ]

                aggregated_reconstruction = self.compressor.batch_decompress(
                    gathered_indices, gathered_values, shape, p
                )
                aggregated_gradient = self.transform.decode(aggregated_reconstruction)

                # 7. Set the parameter's gradient to the aggregated result.
                p.grad.copy_(aggregated_gradient)
                if self.use_sign:
                    p.grad.sign_()

        # 8. Perform the actual optimizer step (e.g., SGD) using the new gradient.
        super().step()
