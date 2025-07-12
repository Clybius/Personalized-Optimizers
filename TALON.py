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

@torch.no_grad()
def orthogonalize(M: torch.Tensor, num_ns_steps=len(NS_COEFFS)) -> torch.Tensor:
    """Orthogonalize a matrix via 5th order Newton-Schulz iteration."""
    transpose = M.shape[0] < M.shape[1]
    if transpose:
        M = M.T
    M = M / (torch.linalg.norm(M) + 1e-12)
    for a, b, c in NS_COEFFS[:num_ns_steps]:
        A = M.T @ M
        I = torch.eye(A.shape[0], dtype=M.dtype, device=M.device)
        M = M @ (a * I + b * A + c * A @ A)
    if transpose:
        M = M.T
    return M

@torch.no_grad()
def _spectral_clip(W: torch.Tensor, sigma_min: float=-1., sigma_max: float=1., ortho_dtype=torch.float32, num_ns_steps=len(NS_COEFFS)):
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
    return result.to(orig_dtype)

@torch.no_grad()
def batch_project(M: torch.Tensor, project_fn: Callable) -> torch.Tensor:
    """Batch project tensors of shape [..., fanout, fanin]. Taken from Modula library."""
    matrix_shape = M.shape[-2:]
    M_flattened = M.reshape(-1, *matrix_shape)

    projected_list = [project_fn(m) for m in M_flattened]
    M_projected = torch.stack(projected_list, dim=0)

    return M_projected.reshape(M.shape) / len(M_flattened)

@torch.no_grad()
def spectral_clip(W: torch.Tensor, sigma_min: float=-1., sigma_max: float=1., ortho_dtype=torch.float32, num_ns_steps=len(NS_COEFFS)):
    return batch_project(W, lambda x: _spectral_clip(x, sigma_min=sigma_min, sigma_max=sigma_max, ortho_dtype=ortho_dtype, num_ns_steps=num_ns_steps))

@torch._dynamo.utils.disable_cache_limit()
@torch.compile(fullgraph=True, mode="reduce-overhead")
def spectral_clip_compiled(W: torch.Tensor, sigma_min: float=-1., sigma_max: float=1., ortho_dtype=torch.float32, num_ns_steps=len(NS_COEFFS)):
    return batch_project(W, lambda x: _spectral_clip(x, sigma_min=sigma_min, sigma_max=sigma_max, ortho_dtype=ortho_dtype, num_ns_steps=num_ns_steps))

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
        invariant (bool):
            Scale the latent into -1 to 1 space via .arctan().sin(), then later divide by the original grad's .arctan().cos(). Its been tested a bit, with the general result of speeding up descent. (default: False).
        spectral_clip (bool):
            Utilize six optimized Newton-Schulz iterations per step to clip the spectral norm to a max of 1. - https://leloykun.github.io/ponder/spectral-clipping/ - https://github.com/leloykun/spectral_clip (default: True).
        spectral_clip_compile (bool):
            Compile the spectral clip function (Highly recommended for a large speed increase). (default: True).
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
        invariant: bool = False,
        spectral_clip: bool = True,
        spectral_clip_compile: bool = True,
        orthograd: bool = False,
        stochastic_fp: bool = True,
    ):

        self._init_lr = lr
        if spectral_clip:
            self.clip_func = spectral_clip_compiled if spectral_clip_compile else spectral_clip

        defaults = dict(
            lr = lr,
            betas = betas,
            weight_decay = weight_decay,
            weight_decay_rate = weight_decay_rate,
            denom_atan2 = denom_atan2,
            invariant = invariant,
            spectral_clip = spectral_clip,
            spectral_clip_compile = spectral_clip_compile,
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
                slow_beta1 = ((betas[1]**(step) - betas[1]) / (betas[1]**(step) - 1.0)) # Short-term bias-correctionless squared EMA beta
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

                # Update value momentum
                value_momentum = value_momentum.mul(slow_beta1).add_(grad.abs(), alpha=1 - slow_beta1)

                # Spectral Clipping / Newton Schulz iters
                dimcount = value_momentum.ndim
                if dimcount > 0 and group["spectral_clip"]:
                    if dimcount > 1:
                        c_t = self.clip_func(value_momentum.mul(sign_momentum))
                    else:
                        c_t = self.clip_func(value_momentum.mul(sign_momentum).view(value_momentum.shape[0], -1)).view(value_momentum.shape)
                else:
                    c_t = value_momentum.mul(sign_momentum)

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
