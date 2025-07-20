# TALON from https://github.com/Clybius/Personalized-Optimizers by Clybius

import torch
from torch.optim import Optimizer
from math import sqrt
from typing import Callable, Tuple
import math

def copy_stochastic_(target: torch.Tensor, source: torch.Tensor):
    # thanks to Nerogar for fast stochastic pytorch implementation
    # https://github.com/pytorch/pytorch/issues/120376#issuecomment-1974828905
    with torch.no_grad():
        # create a random 16 bit integer
        result = torch.randint_like(
            source,
            dtype=torch.int32,
            low=0,
            high=(1 << 16),
        )

        # add the random number to the lower 16 bit of the mantissa
        result.add_(source.view(dtype=torch.int32))

        # mask off the lower 16 bit of the mantissa
        result.bitwise_and_(-65536)  # -65536 = FFFF0000 as a signed int32

        # copy the higher 16 bit into the target tensor
        target.copy_(result.view(dtype=torch.float32))

# Original Spectral Clipping code by leloykun (https://leloykun.github.io/ponder/spectral-clipping/ https://github.com/leloykun/spectral_clip)

"""
@misc{cesista2025spectralclipping,
  author = {Franz Louis Cesista},
  title = {"Fast, Numerically Stable, and Auto-Differentiable Spectral Clipping Via Newton-Schulz Iteration"},
  year = {2025},
  url = {http://leloykun.github.io/ponder/spectral-clipping/},
}
"""
NS_COEFFS = [
    (3.5318, -4.7911, 1.9388),
    (3.3274, -4.0557, 1.5782),
    (3.0809, -3.5160, 1.3464),
    (2.7476, -2.8484, 1.0775),
    (2.2948, -2.0951, 0.7895),
    (2.1535, -1.8338, 0.6869),
]
# New coeffs from https://kexue.fm/archives/11059, may enable later.
"""
NS_COEFFS = [
    (8.287212018145622, -23.59588651909882, 17.300387312530923),
    (4.107059111542197, -2.9478499167379084, 0.54484310829266),
    (3.9486908534822938, -2.908902115962947, 0.5518191394370131),
    (3.3184196573706055, -2.488488024314878, 0.5100489401237208),
    (2.3006520199548186, -1.6689039845747518, 0.4188073119525678),
    (1.8913014077874002, -1.2679958271945908, 0.37680408948524996),
    (1.875, -1.25, 0.375)
]
"""
@torch.no_grad()
def orthogonalize(M: torch.Tensor, num_ns_steps=len(NS_COEFFS), ortho_dtype=None, adaptive=False) -> torch.Tensor:
    """Orthogonalize a matrix via 5th order Newton-Schulz iteration."""
    if ortho_dtype is not None:
        orig_dtype = M.dtype
        M = M.to(ortho_dtype)
    if adaptive:
        M_orig = M.clone()
    transpose = M.shape[0] < M.shape[1]
    if transpose:
        M = M.T
    M = M / (torch.linalg.norm(M) + 1e-20)
    for a, b, c in NS_COEFFS[:num_ns_steps]:
        A = M.T @ M
        I = torch.eye(A.shape[0], dtype=M.dtype, device=M.device)
        M = M @ (a * I + b * A + c * A @ A)
    if transpose:
        M = M.T
    if adaptive:
        M = torch.einsum('ij,ij,ab->ab', M_orig.type_as(M), M, M)
    if ortho_dtype is not None:
        M = M.to(orig_dtype)
    return M

@torch.no_grad()
def block_matmul(
    P1: torch.Tensor, Q1: torch.Tensor, R1: torch.Tensor,
    P2: torch.Tensor, Q2: torch.Tensor, R2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Performs block matrix multiplication elements of the (linear) sub-algebra
    of matrices of the form:
        [P   Q]
        [Q.T R]
    where Q is a MxN matrix, and P and R are symmetric matrices of size MxM and NxN respectively.
    """
    P = P1 @ P2   + Q1 @ Q2.T
    Q = P1 @ Q2   + Q1 @ R2
    R = Q1.T @ Q2 + R1 @ R2
    return P, Q, R

@torch.no_grad()
def newton_schulz_iter(
    P: torch.Tensor, Q: torch.Tensor, R: torch.Tensor,
    a: float, b: float, c: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """5th order blockwise Newton-Schulz iteration for orthogonalization."""
    P2, Q2, R2 = block_matmul(P, Q, R, P, Q, R)
    P4, Q4, R4 = block_matmul(P2, Q2, R2, P2, Q2, R2)
    I_P = a * torch.eye(P.shape[0], dtype=P.dtype, device=P.device)
    I_R = a * torch.eye(R.shape[0], dtype=R.dtype, device=R.device)
    Ppoly = I_P + b * P2 + c * P4
    Qpoly =       b * Q2 + c * Q4
    Rpoly = I_R + b * R2 + c * R4
    return block_matmul(P, Q, R, Ppoly, Qpoly, Rpoly)

@torch.no_grad()
def orthogonalize_blockwise(
    W: torch.Tensor, ortho_dtype=torch.float32, num_ns_steps: int=len(NS_COEFFS)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Orthogonalize a matrix via 5th order blockwise Newton-Schulz iteration.

    Tighter spectral norm bound:
    => Matrices of the form [I_m, W; W.T, I_n] have spectral norm 1 + ||W||_2
    => We can estimate ||W||_2 via power iteration or Gram iteration.
    => However, we can also use the fact that ||W||_2 <= ||W||_F and the latter is much cheaper to compute.

    yeah this is 'translated' from jax to python via gemini
    in the name of PS: 'you can eat my entire ass' or something
    """
    orig_dtype = W.dtype
    m, n = W.shape
    I_m, I_n = torch.eye(m, device=W.device), torch.eye(n, device=W.device)
    # norm = 1 + _power_iterate(W, torch.manual_seed(0), num_iters=16)[1]
    norm = 1 + torch.linalg.norm(W)
    P = (I_m / (norm + 1e-12)).to(ortho_dtype)
    Q = (W   / (norm + 1e-12)).to(ortho_dtype)
    R = (I_n / (norm + 1e-12)).to(ortho_dtype)
    for a, b, c in NS_COEFFS[:num_ns_steps]:
        P, Q, R = newton_schulz_iter(P, Q, R, a=a, b=b, c=c)
    return P.to(orig_dtype), Q.to(orig_dtype), R.to(orig_dtype)

def _spectral_hardcap_blockwise(W: torch.Tensor, sigma_max=1., ortho_dtype=torch.float32, num_ns_steps=len(NS_COEFFS), adaptive=False):
    def _spectral_hardcap_blockwise_util(W: torch.Tensor):
        if adaptive:
            W_orig = W.clone()
        transpose = W.shape[0] > W.shape[1]
        if transpose:
            W = W.T
        orig_dtype = W.dtype
        W = W.to(ortho_dtype)
        # _, Q, R = orthogonalize_blockwise(W, ortho_dtype, num_ns_steps)
        # result = Q + W @ R
        P, Q, _ = orthogonalize_blockwise(W, ortho_dtype, num_ns_steps)
        result = Q + P @ W
        if transpose:
            result = result.T
        if adaptive:
            result = torch.einsum('ij,ij,ab->ab', W_orig.type_as(result), result, result)
        return result.to(orig_dtype)
    return sigma_max * _spectral_hardcap_blockwise_util(W / sigma_max)

def _spectral_clip(W: torch.Tensor, sigma_min: float=-1., sigma_max: float=1., ortho_dtype=torch.float32, num_ns_steps=len(NS_COEFFS), adaptive=False):
    if adaptive:
        W_orig = W.clone()
    flip = W.shape[0] > W.shape[1]
    if flip:
        W = W.T
    orig_dtype = W.dtype
    W = W.to(ortho_dtype)
    OW = orthogonalize(W, num_ns_steps)
    eye_m = torch.eye(W.shape[0], dtype=W.dtype, device=W.device)
    result = (1/2) * (
        (sigma_min + sigma_max) * eye_m
        + (sigma_min * OW - W) @ orthogonalize(sigma_min * OW - W, num_ns_steps).T
        - (sigma_max * OW - W) @ orthogonalize(sigma_max * OW - W, num_ns_steps).T
    ) @ OW
    if flip:
        result = result.T
    if adaptive:
        result = torch.einsum('ij,ij,ab->ab', W_orig.type_as(result), result, result)
    return result.to(orig_dtype)

@torch.no_grad()
def batch_project(M: torch.Tensor, project_fn: Callable) -> torch.Tensor:
    """Batch project tensors of shape [..., fanout, fanin] using vmap."""
    matrix_shape = M.shape[-2:]
    M_flattened = M.reshape(-1, *matrix_shape)

    M_projected = torch.vmap(project_fn)(M_flattened)

    return M_projected.reshape(M.shape) / len(M_flattened)

@torch.no_grad()
def spectral_clip_func(W: torch.Tensor, sigma_min: float=-1., sigma_max: float=1., ortho_dtype=torch.float32, num_ns_steps=len(NS_COEFFS), adaptive=False):
    return batch_project(W, lambda x: _spectral_clip(x, sigma_min=sigma_min, sigma_max=sigma_max, ortho_dtype=ortho_dtype, num_ns_steps=num_ns_steps, adaptive=adaptive))

@torch._dynamo.utils.disable_cache_limit()
@torch.compile(fullgraph=True, mode="reduce-overhead")
def spectral_clip_compiled_func(W: torch.Tensor, sigma_min: float=-1., sigma_max: float=1., ortho_dtype=torch.float32, num_ns_steps=len(NS_COEFFS), adaptive=False):
    return batch_project(W, lambda x: _spectral_clip(x, sigma_min=sigma_min, sigma_max=sigma_max, ortho_dtype=ortho_dtype, num_ns_steps=num_ns_steps, adaptive=adaptive))

@torch.no_grad()
def spectral_hardcap_func(W: torch.Tensor, sigma_min: float=-1., sigma_max: float=1., ortho_dtype=torch.float16, num_ns_steps=len(NS_COEFFS), adaptive=False):
    return batch_project(W, lambda x: _spectral_hardcap_blockwise(x, sigma_max=sigma_max, ortho_dtype=ortho_dtype, num_ns_steps=num_ns_steps, adaptive=adaptive))

@torch._dynamo.utils.disable_cache_limit()
@torch.compile(fullgraph=True, mode="reduce-overhead")
def spectral_hardcap_compiled_func(W: torch.Tensor, sigma_min: float=-1., sigma_max: float=1., ortho_dtype=torch.float16, num_ns_steps=len(NS_COEFFS), adaptive=False):
    return batch_project(W, lambda x: _spectral_hardcap_blockwise(x, sigma_max=sigma_max, ortho_dtype=ortho_dtype, num_ns_steps=num_ns_steps, adaptive=adaptive))

@torch.no_grad()
def orthogonalize_func(W: torch.Tensor, sigma_min: float=-1., sigma_max: float=1., ortho_dtype=torch.float32, num_ns_steps=len(NS_COEFFS), adaptive=False):
    return batch_project(W, lambda x: orthogonalize(x, num_ns_steps=num_ns_steps, ortho_dtype=ortho_dtype, adaptive=adaptive))

@torch._dynamo.utils.disable_cache_limit()
@torch.compile(fullgraph=True, mode="reduce-overhead")
def orthogonalize_compiled_func(W: torch.Tensor, sigma_min: float=-1., sigma_max: float=1., ortho_dtype=torch.float32, num_ns_steps=len(NS_COEFFS), adaptive=False):
    return batch_project(W, lambda x: orthogonalize(x, num_ns_steps=num_ns_steps, ortho_dtype=ortho_dtype, adaptive=adaptive))

@torch.no_grad()
def separate_frequencies(
    grad: torch.Tensor, 
    cutoff_freq_ratio: float = 0.1
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Separates a gradient tensor into low-frequency and high-frequency components
    using the Fast Fourier Transform (FFT).

    Args:
        grad (torch.Tensor): The input gradient tensor. Can be of any shape.
        cutoff_freq_ratio (float): A value between 0.0 and 1.0. It defines the
            radius of the low-pass filter in the frequency domain, as a ratio
            of the smallest dimension size. For example, a value of 0.1 means
            frequencies within a radius of 10% of the smallest dimension size
            are considered "low frequency".

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - low_freq_component (torch.Tensor): The low-frequency part of the gradient.
            - high_freq_component (torch.Tensor): The high-frequency part of the gradient.
    """
    if not 0.0 <= cutoff_freq_ratio <= 1.0:
        raise ValueError("cutoff_freq_ratio must be between 0.0 and 1.0")

    if cutoff_freq_ratio == 1.0:
        return grad.clone(), torch.zeros_like(grad)
    if cutoff_freq_ratio == 0.0:
        return torch.zeros_like(grad), grad.clone()

    # 1. Perform n-dimensional FFT
    grad_fft = torch.fft.fftn(grad)

    # 2. Shift the zero-frequency component to the center for easier masking
    grad_fft_shifted = torch.fft.fftshift(grad_fft)

    # 3. Create a low-pass filter mask
    shape = grad.shape
    # The center of the n-dimensional FFT grid
    center_indices = [s // 2 for s in shape]
    # The radius for the low-pass filter cutoff
    # We use the smallest dimension to define the relative cutoff
    min_dim_size = min(shape)
    cutoff_radius = int(min_dim_size * cutoff_freq_ratio / 2)

    # Create coordinate grids for each dimension
    grid_coords = torch.meshgrid(
        *[torch.arange(s, device=grad.device) for s in shape], 
        indexing='ij'
    )
    
    # Calculate Euclidean distance from the center for each point in the grid
    dist_from_center_sq = torch.zeros_like(grad, dtype=torch.float32)
    for i, center_idx in enumerate(center_indices):
        dist_from_center_sq += (grid_coords[i] - center_idx)**2

    # The mask is True for frequencies within the cutoff radius
    low_pass_mask = dist_from_center_sq <= cutoff_radius**2
    
    # 4. Apply the mask
    low_freq_fft_shifted = grad_fft_shifted * low_pass_mask

    # 5. Inverse shift to move the zero-frequency component back
    low_freq_fft = torch.fft.ifftshift(low_freq_fft_shifted)

    # 6. Perform inverse FFT to get the low-frequency component in the spatial domain
    # The result of ifftn will be complex, but since the input was real, the
    # imaginary part should be negligible. We take the real part.
    low_freq_component = torch.fft.ifftn(low_freq_fft).real

    # 7. The high-frequency component is simply the original gradient minus the low-freq part.
    # This is more numerically stable than performing a second inverse FFT.
    high_freq_component = grad - low_freq_component

    return low_freq_component, high_freq_component

@torch.no_grad()
def freq_sep_func(W: torch.Tensor, cutoff_freq_ratio=0.1):
    return separate_frequencies(W, cutoff_freq_ratio=cutoff_freq_ratio)

def filter_grad(grad, fft_alpha=1.0):
    # 1. Apply n-dimensional FFT
    grad_freq = torch.fft.fftn(grad, dim=list(range(grad.dim())))
    
    # 2. Create a radial low-pass filter
    # Create a grid of frequency coordinates
    freq_dims = [torch.fft.fftfreq(s, device=grad.device) for s in grad.shape]
    # Center the grid for radial calculation
    shifted_freq_dims = [torch.fft.ifftshift(d) for d in freq_dims]
    
    # Create a meshgrid of coordinates
    coords = torch.stack(torch.meshgrid(*shifted_freq_dims, indexing='ij'))
    
    # Calculate the radial distance (L2 norm) from the center (zero frequency)
    # Normalize by the max possible frequency radius for scale invariance
    max_radius = 0.5 * math.sqrt(len(grad.shape))
    radius = torch.linalg.norm(coords, dim=0) / max_radius
    
    # Create a Gaussian low-pass filter.
    # Higher alpha means sharper decay, i.e., more aggressive filtering
    filter_weights = torch.exp(-fft_alpha * (radius ** 2))
    
    # 3. Apply the filter
    filtered_grad_freq = grad_freq * filter_weights
    
    # 4. Apply inverse n-dimensional FFT
    modified_grad = torch.fft.ifftn(filtered_grad_freq, dim=list(range(grad.dim())))
    
    # The result should be real, but take .real to discard negligible imaginary parts
    return modified_grad.real

class TALON(Optimizer):
    r"""
    TALON: Temporal Adaptation via Level and Orientation Normalization. 
    
    Cuts through noise by decoupling the gradient's sign and magnitude into two different momentum states, with a denominator for adaptive learning.

    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.0001).
        betas (float, float, float):
            Coefficient used for computing the sign momentum, running average, and the long-term squared running average (default: 0.9, 0.99, 0.9999999)
        weight_decay (float):
            AdamW-like weight decay, i.e. a L2 penalty (default: 0.0).
        weight_decay_rate (float):
            Decay the multiplier at which rate weight decay is applied, weight_decay * weight_decay_rate**step (default: 0.995).
        denom_atan2 (bool):
            Divide the smooth gradient using .atan2 instead of .div for stability and scale-invariance, removes epsilon/eps - https://arxiv.org/abs/2407.05872 (default: True).
        separate_frequencies (float):
            The ratio of which frequencies to consider "low" before applying the gradient to the model parameters. You can adjust the multiplier of high frequencies via highfreq_mult. (default: 0.0 (recommended 0.1 if used)).
        highfreq_mult (float):
            The multiplier of the separated high frequencies. `separate_frequencies` is disabled by default, so remember to enable it if you intend to change this. (default: 0.1).
        lowpass_grad (float):
            Pre-condition the gradient via a gaussian low-pass filter at this strength, the recommended value is 1.0 or possibly even higher when used. (default: 0.0).
        invariant (bool):
            Scale the latent into -1 to 1 space via .arctan().sin(), then later divide by the original grad's .arctan().cos(). Its been tested a bit, with the general result of speeding up descent. (default: False).
        spectral_clip (bool):
            Utilize six optimized Newton-Schulz iterations per step to clip the spectral norm to a max of 1. - https://leloykun.github.io/ponder/spectral-clipping/ - https://github.com/leloykun/spectral_clip (default: True).
                * Set spectral_min and spectral_max to 0 to enable generic Newton-Schulz orthogonalization.
                * Set spectral_min to any value below -1000.0 to enable block-wise "spectral hardcapping" mode. Likely to be slower in this mode, but more stable.
        spectral_clip_compile (bool):
            Compile the spectral clip function (Highly recommended for a large speed increase). (default: True).
        spectral_min (float):
            The minimum value of the spectral magnitude. Ought to be lower than spectral_max. (default: -1.0).
        spectral_max (float):
            The maximum value of the spectral magnitude. (default: 1.0).
        spectral_adaptive (bool):
            Adapt the result of spectral clipping to adapt to the scale of the gradients - https://github.com/leloykun/adaptive-muon (default: False).
        signscale_power (float):
            Power multiplier for the sign momentum scale. A higher value means more confidence in the sign is needed to scale to the chosen LR, whereas a lower value indicates less confidence is needed. (default: 1.0).
        orthograd (bool):
            Modify the gradient to apply an orthogonal gradient update, - https://arxiv.org/abs/2501.04697 - extended with atan2sin in place of epsilon (default: False).
        stochastic_fp (bool):
            Utilize stochastic rounding for bf16 and fp16 tensors. (default: True).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple = (0.9, 0.99, 1. - 1e-7),
        weight_decay: float = 0.0,
        weight_decay_rate: float = 0.995,
        denom_atan2: bool = True,
        separate_frequencies: float = 0.0,
        highfreq_mult: float = 0.1,
        lowpass_grad: float = 0.0,
        invariant: bool = False,
        spectral_clip: bool = True,
        spectral_clip_compile: bool = True,
        spectral_min: float = -1.,
        spectral_max: float = 1.,
        spectral_adaptive: bool = False,
        signscale_power: float = 1.0,
        orthograd: bool = False,
        stochastic_fp: bool = True,
    ):

        self._init_lr = lr
        if spectral_clip:
            if spectral_min < -1000:
                self.clip_func = spectral_hardcap_compiled_func if spectral_clip_compile else spectral_hardcap_func
            elif spectral_min == 0 and spectral_max == 0:
                self.clip_func = orthogonalize_compiled_func if spectral_clip_compile else orthogonalize_func
            else:
                self.clip_func = spectral_clip_compiled_func if spectral_clip_compile else spectral_clip_func

        defaults = dict(
            lr = lr,
            betas = betas,
            weight_decay = weight_decay,
            weight_decay_rate = weight_decay_rate,
            denom_atan2 = denom_atan2,
            separate_frequencies = separate_frequencies,
            highfreq_mult = highfreq_mult,
            lowpass_grad = lowpass_grad,
            invariant = invariant,
            spectral_clip = spectral_clip,
            spectral_clip_compile = spectral_clip_compile,
            spectral_min = spectral_min,
            spectral_max = spectral_max,
            spectral_adaptive = spectral_adaptive,
            signscale_power = signscale_power,
            orthograd = orthograd,
            stochastic_fp = stochastic_fp,
        )

        super(TALON, self).__init__(params, defaults)

    @torch.no_grad()
    def orthograd_atan2sin(self, p, grad):
        w = p.view(-1)
        g = grad.view(-1)

        dot_product = torch.dot(w, g).atan2_(torch.dot(w, w))
        sin_dot_product = torch.sin(dot_product)

        g_atansin = g.to(dtype=torch.float32, copy=True).atan().sin_()
        g_atancos = g.to(dtype=torch.float32, copy=True).atan().cos_()

        g_orth = g_atansin.sub(w.atan().sin_(), alpha=sin_dot_product).div(g_atancos)

        g_orth_scaled = g_orth.mul(g.norm(2).div_(g_orth.norm(2).clamp_min_(1e-16)))

        grad.copy_(g_orth_scaled.view_as(grad))
    
    @torch.no_grad()
    def invariance(self, grad, degrad = None):
        if degrad is None:
            g_atansin = grad.atan().sin_()
            g_atancos = grad.atan().cos_()

            return g_atansin, g_atancos
        else:
            return grad.atan2(degrad).mul_(1.27323954474)

    @torch.no_grad()
    def reset(self):
        pass

    @torch.no_grad()
    def step(self, closure = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            lr = group["lr"]
            betas = group["betas"]
            weight_decay = group["weight_decay"]
            weight_decay_rate = group["weight_decay_rate"]
            step = group['step']

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]

                grad = p.grad.data

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["value_momentum"] = torch.ones_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["stage2_emasq"] = torch.ones_like(p.data)
                    state["sign_momentum"] = torch.zeros_like(grad)

                # Detach
                p_fp32 = p.detach().clone()
                value_momentum = state["value_momentum"].detach().clone()
                stage2_emasq = state["stage2_emasq"].detach().clone()
                sign_momentum = state["sign_momentum"].detach().clone()

                # Unpack
                if p.dtype in {torch.float16, torch.bfloat16} and group["stochastic_fp"]:
                    grad = grad.to(torch.float32)
                    value_momentum = state['value_momentum'].detach().clone().to(torch.float32)
                    stage2_emasq = state['stage2_emasq'].detach().clone().to(torch.float32)
                    sign_momentum = state['sign_momentum'].detach().clone().to(torch.float32)
                    p_fp32 = p.detach().clone().to(torch.float32)

                # Create betas
                slow_beta1 = ((betas[1]**(step) - betas[1]) / (betas[1]**(step) - 1.0)) # Bias-correctionless value momentum/EMA beta
                slow_beta2 = ((betas[2]**(step) - betas[2]) / (betas[2]**(step) - 1.0)) # Long-term bias-correctionless squared EMA beta

                # Absmax clip value for early stability
                clip_lambda = step**0.25

                # RMS clip for stability with denom
                rms = grad.pow(2).mean().sqrt_().clamp_min_(1)
                grad = grad.div(rms)

                # Orthograd
                if group["orthograd"] and p_fp32.data.nelement() > 1: # Might just be me, but I've had the most success via ndim > 1
                    self.orthograd_atan2sin(p_fp32, grad)

                # Update sign momentum
                sign_momentum = sign_momentum.lerp(grad.sign(), weight=1. - betas[0])

                # Clip grad to prevent INF
                grad = torch.where(
                    grad.abs() > 255,
                    grad.mul(255 / grad.abs()),
                    grad
                )

                dimcount = grad.ndim
                if dimcount > 0 and group["lowpass_grad"] != 0:
                    grad = filter_grad(grad, fft_alpha=group["lowpass_grad"]).abs().mul_(grad.sign())

                # Update value momentum
                value_momentum = value_momentum.mul(slow_beta1).add_(grad.abs(), alpha=1 - slow_beta1)

                # Spectral Clipping / Newton Schulz iters
                if dimcount > 0 and group["spectral_clip"]:
                    if dimcount > 1:
                        c_t = self.clip_func(value_momentum, sigma_min=group["spectral_min"], sigma_max=group["spectral_max"], adaptive=group["spectral_adaptive"])
                    else:
                        c_t = self.clip_func(value_momentum.view(len(value_momentum), -1), sigma_min=group["spectral_min"], sigma_max=group["spectral_max"], adaptive=group["spectral_adaptive"]).view(value_momentum.shape)
                else:
                    c_t = value_momentum

                # Invariant (Stage 1)
                if group["invariant"] and c_t.nelement() > 0:
                    c_t, degrad = self.invariance(c_t)

                # Denom
                if group["denom_atan2"]:
                    full_step = c_t.atan2(stage2_emasq.sqrt()).mul_(1.27323954474)
                else:
                    stage2_denom = torch.clamp(stage2_emasq.sqrt(), 1e-16)
                    full_step = c_t.div(stage2_denom).clamp_(-clip_lambda, clip_lambda)

                # ADOPT-style update squared momentum
                stage2_emasq = stage2_emasq.mul(slow_beta2).addcmul_(grad, grad, value=1 - slow_beta2)

                # Invariant (Stage 2)
                if group["invariant"] and grad.nelement() > 0:
                    full_step = self.invariance(full_step, degrad)

                if dimcount > 0 and group["separate_frequencies"] != 0:
                    lf_grad, hf_grad = freq_sep_func(full_step, cutoff_freq_ratio=group["separate_frequencies"])
                    full_step = (lf_grad + hf_grad.mul(group["highfreq_mult"]))

                # Apply sign and the confidence/scale in the sign.
                full_step = full_step.mul(sign_momentum.abs().pow_(group["signscale_power"]).mul_(sign_momentum.sign()))

                # Perform weight decay
                if weight_decay != 0:
                    grad_weights = p_fp32.data

                    full_step = full_step.add(grad_weights, alpha=weight_decay * weight_decay_rate**group["step"])

                p_fp32.data.add_(full_step, alpha=-lr)
                if p.dtype in {torch.float16, torch.bfloat16} and group["stochastic_fp"]:
                    copy_stochastic_(state["value_momentum"], value_momentum)
                    copy_stochastic_(state["stage2_emasq"], stage2_emasq)
                    copy_stochastic_(state["sign_momentum"], sign_momentum)
                    copy_stochastic_(p, p_fp32)
                else:
                    state["value_momentum"].copy_(value_momentum)
                    state["stage2_emasq"].copy_(stage2_emasq)
                    state["sign_momentum"].copy_(sign_momentum)
                    p.copy_(p_fp32)
        return loss
