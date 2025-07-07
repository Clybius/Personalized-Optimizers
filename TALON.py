# TALON from https://github.com/Clybius/Personalized-Optimizers by Clybius

import torch
from torch.optim import Optimizer
from math import sqrt
from enum import IntEnum
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

# https://github.com/kozistr/pytorch_optimizer/blob/6397d56279ad80b26c4bba7fb4b04852b517fdeb/pytorch_optimizer/optimizer/shampoo_utils.py#L533
def zero_power_via_newton_schulz_6(
    g: torch.Tensor, eps: float = 1e-16
) -> torch.Tensor:
    r"""Compute the zeroth power / orthogonalization of G.

    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a quintic iteration
    whose coefficients are selected to maximize the slope at zero. For the purpose of minimizing steps, it turns out
    to be empirically effective to keep increasing the slope at zero even beyond the point where the iteration no
    longer converges all the way to one everywhere on the interval. This iteration therefore does not produce UV^T but
    rather something like US'V^T where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt
    model performance at all relative to UV^T, where USV^T = G is the SVD.

    :param g: torch.Tensor. matrix.
    :param num_steps: int. number of iterations.
    :param eps: float. add this times I to G, to make is positive definite. For scaling, we multiply it by the largest
        eigenvalue of G.
    :param weights: Tuple[int, int, int]. weights.
    """
    if len(g.shape) != 2:
        raise ValueError('shape of g must be 2-dimensional')

    abc_list = [
      (3955/1024, -8306/1024, 5008/1024),
      (3735/1024, -6681/1024, 3463/1024),
      (3799/1024, -6499/1024, 3211/1024),
      (4019/1024, -6385/1024, 2906/1024),
      (2677/1024, -3029/1024, 1162/1024),
      (2172/1024, -1833/1024,  682/1024)
   ]

    x = g.float()
    x = x.div(x.norm().add_(eps))

    if g.size(0) > g.size(1):
        x = x.T

    for weight in abc_list:
        a = x @ x.T
        b = weight[1] * a + weight[2] * a @ a
        x = weight[0] * x + b @ x

    if g.size(0) > g.size(1):
        x = x.T

    x = torch.einsum('ij,ij,ab->ab', g.type_as(x), x, x)

    return x

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
        adaptive_muon (bool):
            Utilize six optimized Newton-Schulz iterations per step to compute the orthogonalization of the gradient, and adapt to the gradient norm - https://arxiv.org/abs/2410.21265 - https://github.com/leloykun/adaptive-muon (default: True).
        orthograd (bool):
            Modify the gradient to apply an orthogonal gradient update, - https://arxiv.org/abs/2501.04697 - extended with atan2 in place of epsilon - https://arxiv.org/abs/2407.05872 (default: False).
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
        adaptive_muon: bool = True,
        orthograd: bool = False,
        stochastic_fp: bool = True,
    ):

        self._init_lr = lr

        defaults = dict(
            lr = lr,
            betas = betas,
            weight_decay = weight_decay,
            weight_decay_rate = weight_decay_rate,
            denom_atan2 = denom_atan2,
            invariant = invariant,
            adaptive_muon = adaptive_muon,
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

                rms = grad.pow(2).mean().sqrt_().clamp_min_(1)
                grad = grad.div(rms)

                # Orthograd
                if group["orthograd"] and p_fp32.data.nelement() > 1: # Might just be me, but I've had the most success via ndim > 1
                    self.orthograd_atan2sin(p_fp32, grad)

                # Update sign momentum
                sign_momentum = sign_momentum.lerp(grad.sign(), weight=1. - betas[0])

                # Adaptive Muon / Newton Schulz iters
                if grad.ndim > 0 and group["adaptive_muon"]:
                    grad = zero_power_via_newton_schulz_6(grad.view(len(grad), -1)).view(grad.shape)

                # Clip grad to prevent INF
                grad = torch.where(
                    grad.abs() > 255,
                    grad.mul(255 / grad.abs()),
                    grad
                )

                # ADOPT-style update squared momentum (Stage 1)
                value_momentum = value_momentum.mul(slow_beta1).add_(grad.abs(), alpha=1 - slow_beta1)

                # Denom (Stage 1)
                c_t = value_momentum.mul(sign_momentum)

                if group["invariant"] and c_t.nelement() > 0:
                    c_t, degrad = self.invariance(c_t)

                # Denom (Stage 2)
                if group["denom_atan2"]:
                    full_step = c_t.atan2(stage2_emasq.sqrt()).mul_(1.27323954474)
                else:
                    stage2_denom = torch.clamp(stage2_emasq.sqrt(), 1e-16)
                    full_step = c_t.div(stage2_denom).clamp_(-clip_lambda, clip_lambda)

                # ADOPT-style update squared momentum (Stage 2)
                stage2_emasq = stage2_emasq.mul(slow_beta2).addcmul_(grad, grad, value=1 - slow_beta2)

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