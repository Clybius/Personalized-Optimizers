# SCGOpt from https://github.com/Clybius/Personalized-Optimizers by Clybius

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
        M = (M_orig * M).sum() * M
    if ortho_dtype is not None:
        M = M.to(orig_dtype)
    return M

@torch.no_grad()
def orthogonalize_func(W: torch.Tensor, sigma_min: float=-1., sigma_max: float=1., ortho_dtype=torch.float32, num_ns_steps=len(NS_COEFFS), adaptive=False):
    return orthogonalize(W, num_ns_steps=num_ns_steps, ortho_dtype=ortho_dtype, adaptive=adaptive)

@torch._dynamo.utils.disable_cache_limit()
@torch.compile(fullgraph=True, mode="reduce-overhead")
def orthogonalize_compiled_func(W: torch.Tensor, sigma_min: float=-1., sigma_max: float=1., ortho_dtype=torch.float32, num_ns_steps=len(NS_COEFFS), adaptive=False):
    return orthogonalize(W, num_ns_steps=num_ns_steps, ortho_dtype=ortho_dtype, adaptive=adaptive)

def filter_grad(grad, fft_alpha=1.0):
    # 1. Apply n-dimensional FFT
    grad_freq = torch.fft.fftn(grad, norm='ortho')
    
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
    modified_grad = torch.fft.ifftn(filtered_grad_freq, norm='ortho')
    
    # The result should be real, but take .real to discard negligible imaginary parts
    return modified_grad.real

def create_gaussian_mask(shape, sigma=1.0, device='cpu'):
    """
    Creates a n-dimensional Gaussian mask, centered for use with fftshift.
    """
    freq_dims = [torch.fft.fftfreq(s, device=device) for s in shape]
    # Center the grid for radial calculation
    shifted_freq_dims = [torch.fft.ifftshift(d) for d in freq_dims]
    
    # Create a meshgrid of coordinates
    coords = torch.stack(torch.meshgrid(*shifted_freq_dims, indexing='ij'))
    
    # Calculate the radial distance (L2 norm) from the center (zero frequency)
    # Normalize by the max possible frequency radius for scale invariance
    max_radius = 0.5 * math.sqrt(len(shape))
    radius = torch.linalg.norm(coords, dim=0) / max_radius
    
    # Create a Gaussian low-pass filter.
    # Higher alpha means sharper decay, i.e., more aggressive filtering
    filter_weights = torch.exp(-sigma * (radius ** 2))
    return filter_weights

def similarity_fft(grad, prev_grad, sigma=0.0):
    # 1. Apply n-dimensional FFT
    grad_freq = torch.fft.fftn(grad, norm='ortho')
    prev_grad_freq = torch.fft.fftn(prev_grad, norm='ortho')

    grad_freq_shifted = torch.fft.fftshift(grad_freq)
    prev_grad_freq_shifted = torch.fft.fftshift(prev_grad_freq)

    agreement_mask = grad_freq_shifted.abs() * prev_grad_freq_shifted.abs().conj()

    mask_max = torch.max(agreement_mask.abs())
    if mask_max > 1e-16:
        agreement_mask /= mask_max
    
    new_grad_fft = grad_freq_shifted * agreement_mask.real

    if sigma != 0:
        gaussian_mask = create_gaussian_mask(grad.shape, sigma=sigma, device=grad.device)
        new_grad_fft = new_grad_fft * gaussian_mask

    new_grad_fft = torch.fft.ifftshift(new_grad_fft)

    new_grad = torch.fft.ifftn(new_grad_fft, norm='ortho').real

    return new_grad

class SCGOpt(Optimizer):
    r"""
    SCGOpt: Sign-based Centralized Gradient Optimization.

    Separates momentum states into full gradient and centralized gradient for smoother and faster descent, with a few extra features for boosting and stabilizing descent.

    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.0001).
        betas (float, float, float):
            Coefficient used for computing the centralized momentum, full gradient momentum (used for centering), and the long-term squared running average (default: 0.95, 0.9999999, 0.9999999).
        weight_decay (float):
            AdamW-like weight decay, i.e. a L2 penalty (default: 0.0).
        weight_decay_rate (float):
            Decay the multiplier at which rate weight decay is applied, weight_decay * weight_decay_rate**step (default: 0.995).
        centralization (float):
            Subtract the full gradient momentum from the current gradient at this ratio (default: 1.0).
        spectral_clip (bool):
            Utilize six optimized Newton-Schulz iterations per step to clip the spectral norm to a max of 1. - https://leloykun.github.io/ponder/spectral-clipping/ - https://github.com/leloykun/spectral_clip (default: False).
        spectral_adaptive (bool):
            Adapt the result of spectral clipping to adapt to the scale of the gradients - https://github.com/leloykun/adaptive-muon (default: True).
        spectral_clip_compile (bool):
            Compile the spectral clip function (Highly recommended for a large speed increase) (default: True).
        spectral_clip_dtype (torch.dtype in string format):
            Sets the dtype of spectral clipping calculation. Recommended to use torch.float32 (or leave at default of None) (default: None, which results in torch.float32).
        adaptive (bool):
            Scale the full step to the momentumized average gradient, always utilizes RMS normalization on the gradient if True, otherwise caps RMS at 1.0 (default: True).
        adaptive_min (float):
            Minimum multiplier for the adaptive scale (default: -1.0).
        adaptive_max (float):
            Maximum multiplier for the adaptive scale (default: 1.0).
        use_sign (bool):
            Transform the gradient into its .sign() based form (-1 if negative or 1 if positive). May be more stable in noisy scenarios (default: True).
        lowpass_grad (float):
            Pre-conditions the gradient via a low-pass filter that maintains the direction of the gradient. Higher = stronger filtering, 0 = disabled (default: 0.0).
        sim_match (bool):
            Filters the frequencies of the running average with the gradient of the current step's frequencies (default: False).
        cautious_min (float):
            A value other than 1.0 will utilize cautious-stepping. At 0.0, this zeros out parts of the momentum which don't correlate with the current gradient's direction. 0.5 will halve it instead (default: 0.0).
        stochastic_fp (bool):
            Utilize stochastic rounding for bf16 and fp16 tensors. (default: True).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: float = (0.95, 0.9999999, 0.9999999),
        weight_decay: float = 0.0,
        weight_decay_rate: float = 0.998,
        centralization: float = 1.0,
        spectral_clip: bool = False,
        spectral_adaptive: bool = True,
        spectral_clip_compile: bool = True,
        spectral_clip_dtype = None, # Can be set to torch.bfloat16, torch.float16, torch.float32, or even torch.float64 if you're insane in the membrane.
        adaptive: bool = True,
        adaptive_min: float = -1.,
        adaptive_max: float = 1.,
        use_sign: bool = True,
        lowpass_grad: float = 0.0,
        sim_match: bool = False,
        cautious_min: float = 0.0,
        stochastic_fp: bool = True,
    ):

        self._init_lr = lr

        if spectral_clip:
            self.clip_func = orthogonalize_compiled_func if spectral_clip_compile else orthogonalize_func

        if spectral_clip_dtype is None:
            spectral_clip_dtype = torch.float32

        if isinstance(spectral_clip_dtype, str):
            dtype_name = spectral_clip_dtype.split('.')[-1] # Gets "float16"
            spectral_clip_dtype = getattr(torch, dtype_name)

        defaults = dict(
            lr = lr,
            betas = betas,
            weight_decay = weight_decay,
            weight_decay_rate = weight_decay_rate,
            centralization = centralization,
            spectral_clip = spectral_clip,
            spectral_adaptive = spectral_adaptive,
            spectral_clip_compile = spectral_clip_compile,
            spectral_clip_dtype = spectral_clip_dtype,
            adaptive = adaptive,
            adaptive_min = adaptive_min,
            adaptive_max = adaptive_max,
            use_sign = use_sign,
            lowpass_grad = lowpass_grad,
            sim_match = sim_match,
            cautious_min = cautious_min,
            stochastic_fp = stochastic_fp,
        )

        super(SCGOpt, self).__init__(params, defaults)

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
            beta, beta2, beta3 = group["betas"][0], group["betas"][1], group["betas"][2]
            weight_decay = group["weight_decay"]
            weight_decay_rate = group["weight_decay_rate"]
            centralization = group["centralization"]

            step = group['step']

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]

                grad = p.grad.data

                dimcount = grad.ndim

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["denom"] = torch.ones_like(grad).mean() if group["use_sign"] else torch.ones_like(grad)
                    state["value_momentum"] = torch.zeros_like(grad)
                    state["centralized_momentum"] = torch.zeros_like(grad)

                # Detach
                p_fp32 = p.detach().clone()
                denom = state["denom"].detach().clone()
                value_momentum = state["value_momentum"].detach().clone()
                centralized_momentum = state["centralized_momentum"].detach().clone()

                # Unpack
                if p.dtype in {torch.float16, torch.bfloat16} and group["stochastic_fp"]:
                    grad = grad.to(torch.float32)
                    denom = state['denom'].detach().clone().to(torch.float32)
                    value_momentum = state['value_momentum'].detach().clone().to(torch.float32)
                    centralized_momentum = state['centralized_momentum'].detach().clone().to(torch.float32)
                    p_fp32 = p.detach().clone().to(torch.float32)

                # Averaged beta (step 1 = 0, step 2 = 0.5, step 3 = 0.6667, step 4 = 0.75...)
                slow_beta2 = ((beta2**(step) - beta2) / (beta2**(step) - 1.0))
                slow_beta3 = ((beta3**(step) - beta3) / (beta3**(step) - 1.0))

                # ADOPT-style clamping for early stability / to prevent NaNs
                grad = grad.clamp(-step, step)

                if group["use_sign"]:
                    grad = grad.sign()
                else:
                    # Low-pass filter via FFT, maintains direction
                    if dimcount > 0 and group["lowpass_grad"] != 0:
                        grad = filter_grad(grad, fft_alpha=group["lowpass_grad"]).abs().mul_(grad.sign())

                    # Move RMS to 1.0 if scale-matching (adaptive) is enabled, otherwise cap the RMS at 1.0 if not
                    if group["adaptive"]:
                        rms = grad.pow(2).mean().sqrt_().clamp_min_(1e-16) # Cap at RMS of 1.0
                    else:
                        rms = grad.pow(2).mean().sqrt_().clamp_min_(1.0) # Cap at min RMS of 1.0
                    grad = grad.div(rms)

                # ADOPT-style denominator update (un-updated denom)
                current_denom = denom.sqrt()

                # Centralize gradient by removing running average
                centralized_grad = grad.sub(value_momentum, alpha=centralization)#.mul(value_momentum.sum().clamp(group["adaptive_min"], group["adaptive_max"])) # Nesterov

                # Momentumize the centralized gradient
                centralized_momentum = centralized_momentum.lerp(centralized_grad, weight=1. - beta)

                # Update full momentum
                value_momentum = value_momentum.lerp(grad, weight=1. - slow_beta2)

                # Add back full momentum to the centralized gradient
                exp_avg = centralized_grad.lerp(centralized_momentum, weight=beta).add_(grad.lerp(value_momentum, weight=slow_beta2), alpha=centralization)

                # Update denominator with either centralized gradient, or its mean when utilizing a sign-based gradient
                denom = denom.lerp(centralized_grad.pow(2).mean() if group["use_sign"] else centralized_grad.pow(2), weight=1. - slow_beta3)

                # Frequency matching the momentumized update with the current step's gradient
                if dimcount > 0 and group["sim_match"] and not group["use_sign"]:
                    exp_avg = similarity_fft(exp_avg, grad)

                # Spectral Clipping / Newton Schulz iters
                if dimcount >= 1 and group["spectral_clip"]:
                    if dimcount > 2:
                        exp_avg_2d = exp_avg.reshape(len(exp_avg), -1) # Make 2D if conv or 1 dim
                    elif dimcount < 2:
                        exp_avg_2d = exp_avg.reshape(1, -1) # Make 2D if conv or 1 dim
                    else:
                        exp_avg_2d = exp_avg

                    flip = exp_avg_2d.shape[0] > exp_avg_2d.shape[1]
                    if flip:
                        exp_avg_2d = exp_avg_2d.T # Flip if first dim is larger

                    exp_avg_2d = self.clip_func(exp_avg_2d, sigma_min=0., sigma_max=0., adaptive=group["spectral_adaptive"], ortho_dtype=group["spectral_clip_dtype"])

                    if flip:
                        exp_avg_2d = exp_avg_2d.T

                    exp_avg = exp_avg_2d.view_as(exp_avg)

                # Cautious update (zero-out update where the update isn't in the direction of the current gradient)
                scale_factor_mask = torch.where(grad * exp_avg > 0, torch.ones_like(exp_avg), torch.ones_like(exp_avg) * group["cautious_min"]).to(exp_avg.dtype)
                scale_factor_mask = scale_factor_mask.div(scale_factor_mask.mean().clamp_min_(1e-3))

                # Atan2-Adam denominator for scale invariance
                full_step = exp_avg.mul(scale_factor_mask).atan2(current_denom).mul_(1.27323954474)
                # Scale the full step with the gradient
                if group["adaptive"] and dimcount > 0:
                    scale_factor = (exp_avg * full_step).sum().clamp(group["adaptive_min"], group["adaptive_max"])

                    full_step = scale_factor * full_step

                # Perform weight decay
                if weight_decay != 0:
                    grad_weights = p_fp32.data

                    full_step = full_step.add(grad_weights, alpha=weight_decay * weight_decay_rate**group["step"])

                p_fp32.data.add_(full_step, alpha=-lr)

                # Stochastic update
                if p.dtype in {torch.float16, torch.bfloat16} and group["stochastic_fp"]:
                    copy_stochastic_(state["denom"], denom)
                    copy_stochastic_(state["value_momentum"], value_momentum)
                    copy_stochastic_(state["centralized_momentum"], centralized_momentum)
                    copy_stochastic_(p, p_fp32)
                else:
                    state["denom"].copy_(denom)
                    state["value_momentum"].copy_(value_momentum)
                    state["centralized_momentum"].copy_(centralized_momentum)
                    p.copy_(p_fp32)
        return loss