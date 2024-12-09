# FMARSCropV2 from https://github.com/Clybius/Personalized-Optimizers by Clybius
# New additions such as stochastic rounding, adaptive eps, among other improvements implemented by Machina.
import torch
from torch.optim import Optimizer

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

# From pytorch_optimizer: https://github.com/kozistr/pytorch_optimizer
def unit_norm(x: torch.Tensor, norm: float = 2.0) -> torch.Tensor:
    r"""Get norm of unit."""
    keep_dim = True
    dim = None

    x_len: int = len(x.shape)
    if x_len <= 1:
        keep_dim = False
    elif x_len in (2, 3):
        dim = 1
    elif x_len == 4:
        dim = (1, 2, 3)
    else:
        dim = tuple(range(1, x_len))

    return x.norm(p=norm, dim=dim, keepdim=keep_dim)

def agc_global_norm(p: torch.Tensor, grad: torch.Tensor, agc_eps: float, agc_clip_val: float, eps: float = 1e-6) -> torch.Tensor:
    r"""Clip gradient values based on the global norm.
    Scale the entire gradient tensor if its norm exceeds a threshold.

    References:
        [Brock, Smith, De, Simonyan 2021] High-Performance Large-Scale Image
        Recognition Without Normalization.

    :param p: torch.Tensor. Parameter tensor.
    :param grad: torch.Tensor. Gradient tensor.
    :param agc_eps: float. A small epsilon value to prevent division by zero.
    :param agc_clip_val: float. Clipping threshold multiplier.
    :param eps: float. Small value to prevent division by zero in normalization.
    """
    # Compute the global norm of the parameters and gradients
    p_norm = unit_norm(p).clamp_(min=agc_eps)
    g_norm = unit_norm(grad)

    # Compute the maximum allowed norm for the gradients
    max_norm = p_norm * agc_clip_val

    clipped_grad = grad * (max_norm / g_norm.clamp_min_(eps))

    return torch.where(g_norm > max_norm, clipped_grad, grad)

class FMARSCropV2(Optimizer):
    r"""
    FMARSCropV2: Fisher-accelerated MARS (https://arxiv.org/abs/2411.10438), with momentum-based Compass-style amplification, with customized ADOPT AdamW changes (https://arxiv.org/abs/2411.02853), and cautious stepping.
    Un-official MARS implementation is credited to Less Wright (lessw2020).
    Intended to arrive at the minima faster and in a more stable manner than FMARSCrop_ExMachina and V1
    Thanks to Machina for introducing the usage of stochastic rounding, adaptive_eps, and further testing!
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.0001).
        betas (float, float):
            coefficients used for computing running averages of
            gradient difference FIM and approx. natural grad FIM (default: 0.999, 0.9999).
        eps (float):
            Term the denominator is minimally clamped to, to
            improve numerical stability. (default: 1e-6).
        eps2 (float):
            Term to multiple the RMS of the grad to calculate adaptive eps. (default: 1e-2).
        eps_floor (float):
            Term to set a floor for the eps, to prevent NaNs. (default: None, disabling adaptive eps. If 0, round to 1e-36).
        weight_decay (float):
            AdamW-like weight decay, i.e. a L2 penalty (default: 0.01).
        centralization (float):
            Center model grad (default: 0.0).
        moment_centralization (float):
            Center the slow momentum / EMA - https://arxiv.org/abs/2207.09066 (default: 0.0).
        diff_mult (float):
            Multiplier for difference amplification, adds another memory state (slightly increased VRAM usage) (default: 0.0).
        momentum_beta (float):
            Beta value for slow momentum / EMA (default: 0.99) (Alternative recommendation: 0.9999).
        momentum_lambda (float):
            Amplification exponent for slow momentum / EMA (default: 0.1) (Alternative recommendation: 0.25).
        gamma (float):
            Scaling parameter for gradient correction for MARS - https://arxiv.org/abs/2411.10438 (default: 0.001).
        clip (float):
            Value to clip the grad's RMS at (default: 1.0).
        adaptive_clip (float):
            Adaptive clip value to apply to the corrected gradient, before further use by the optimizer. (default: 1.0).
        adaptive_clip_eps (float):
            The eps for adaptive gradient clipping, provides a minimum to avoid parameters 
            not getting updating due to very small gradients being clipped excessively. (default: 1e-3).
        cautious (bool):
            Use cautious mask on parameter update - https://arxiv.org/abs/2411.16085 (default: True).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas = (0.999, 0.9999),
        eps: float = 1e-6,
        eps2: float = 1e-2,
        eps_floor: float = None,
        weight_decay: float = 0.01,
        centralization: float = 0.0,
        moment_centralization: float = 0.0,
        diff_mult: float = 0.0,
        momentum_beta: float = 0.99,
        momentum_lambda: float = 0.1,
        gamma: float = 0.001,
        clip: float = 1.0,
        adaptive_clip: float = 1.0,
        adaptive_clip_eps: float = 1e-3,
        cautious: bool = True,
    ):

        # Override zero to 1e-36, as zero and float32.tiny NaNs
        if eps_floor is not None and eps_floor < eps and eps_floor <= 0:
            eps_floor = 1e-36

        defaults = dict(
            lr = lr,
            betas = betas,
            eps = eps,
            eps2 = eps2,
            eps_floor = eps_floor,
            weight_decay = weight_decay,
            centralization = centralization,
            moment_centralization = moment_centralization,
            diff_mult = diff_mult,
            momentum_beta = momentum_beta,
            momentum_lambda = momentum_lambda,
            gamma = gamma,
            clip = clip,
            adaptive_clip = adaptive_clip,
            adaptive_clip_eps = adaptive_clip_eps,
            cautious = cautious,
        )

        super(FMARSCropV2, self).__init__(params, defaults)

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group["params"]:
                state = self.state[p]

                state["fim"] = torch.ones_like(p.data)
                # Fisher information matrix
                state["momentum"] = torch.zeros_like(p.data)
                # Prev grad
                state["prev_grad"] = torch.zeros_like(p.data).detach()
                if group["diff_mult"] > 0:
                    state["grad_diff_fim"] = torch.ones_like(p.data)

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

            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            centralization = group["centralization"]
            moment_centralization = group["moment_centralization"]
            diff_mult = group["diff_mult"]
            momentum_beta = group["momentum_beta"]
            momentum_lambda = group["momentum_lambda"]
            gamma = group["gamma"]
            clip = group["clip"]
            step = group['step']
            eps = group["eps"]
            eps2 = group["eps2"]
            eps_floor = group["eps_floor"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["momentum"] = torch.zeros_like(p)
                    state["fim"] = torch.ones_like(p)
                    state["prev_grad"] = -p.grad.clone().to(p.dtype).detach()
                    if diff_mult > 0:
                        state["grad_diff_fim"] = torch.ones_like(p)

                state = self.state[p]

                grad = p.grad

                p_fp32 = p

                prev_grad = state["prev_grad"]
                fim = state["fim"]
                momentum = state["momentum"]

                # Unpack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.to(torch.float32)
                    fim = state["fim"].to(torch.float32)
                    momentum = state["momentum"].to(torch.float32)
                    prev_grad = state["prev_grad"].to(torch.float32)
                    p_fp32 = p.clone().to(torch.float32)

                prev_grad = prev_grad.add(grad)

                # Calculate cₜ (gradient with correction term)
                correction = gamma * beta1 / (1 - beta1) * prev_grad
                c_t = grad + correction

                # Gradient clipping (if necessary)
                if group["adaptive_clip"] > 0.0:
                    c_t.copy_(agc_global_norm(p_fp32, c_t, group["adaptive_clip_eps"], group["adaptive_clip"]))

                clip_lambda = step**0.25

                fim_slow_beta = ((beta2**step - beta2) / (beta2**step - 1.0)) ** (1/2)

                if eps_floor is not None and eps_floor < eps:
                    rms_grad = grad.pow(2).mean().sqrt_()
                    curr_eps = max(min(eps, eps2 * rms_grad.item()), eps_floor) # Set a floor for eps to avoid NaN
                else:
                    curr_eps = eps

                if diff_mult > 0:
                    # Get previous grad, initialized at 0 (first step is just grad)
                    # grad_diff will contain the difference between prev grad and current grad
                    grad_diff = prev_grad * diff_mult

                    rms = grad_diff.pow(2).mean().sqrt_()
                    divisor = max(clip, rms) / clip
                    grad_diff.div_(divisor)

                    grad_diff_fim = state["grad_diff_fim"]

                    # Unpack
                    if p.dtype in {torch.float16, torch.bfloat16}:
                        grad_diff_fim = state["grad_diff_fim"].to(torch.float32)

                    # Get natural gradient (squared ema, obtained sqrt of ema)
                    diff_fim_base = torch.clamp(grad_diff_fim.sqrt(), curr_eps)

                    grad_diff_fim.mul_(beta1).addcmul_(grad_diff, grad_diff, value=1 - beta1).clamp_(-clip_lambda, clip_lambda)

                    # pack
                    if p.dtype in {torch.float16, torch.bfloat16}:
                        copy_stochastic_(state["grad_diff_fim"], grad_diff_fim)
                else:
                    diff_fim_base = 1.0

                approx_grad_nat = c_t.div(diff_fim_base)
                rms = approx_grad_nat.pow(2).mean().sqrt_()
                divisor = max(clip, rms) / clip
                approx_grad_nat.div_(divisor)

                fim_base = torch.clamp(fim.sqrt(), curr_eps)

                grad_nat = approx_grad_nat.div(fim_base).div_(diff_fim_base)
                rms = grad_nat.pow(2).mean().sqrt_()
                divisor = max(clip, rms) / clip
                grad_nat.div_(divisor)

                momentum.mul_(momentum_beta).add_(grad_nat, alpha=1 - momentum_beta)

                # Compass-style amplification
                if moment_centralization != 0:
                    momentum_cent = momentum - torch.mean(momentum) * moment_centralization
                else:
                    momentum_cent = momentum
                # Apply full step
                if group['cautious']:
                    # Apply caution as per 'Cautious Optimizers' - https://arxiv.org/abs/2411.16085
                    mask = (momentum_cent * grad_nat < 0).to(momentum_cent.dtype) # Unsure if disagreement masking is more useful than agreement masking, or not masking at all.
                    mask.div_(mask.mean().clamp_(min=1e-3)) #                       It should theoretically help prevent poor updates?
                    momentum_cent = momentum_cent * mask
                full_step = grad_nat.add(momentum_cent, alpha=step**momentum_lambda)

                # center the gradient vector
                if centralization != 0 and full_step.dim() > 1:
                    full_step.sub_(
                        full_step.mean(dim=tuple(range(1, full_step.dim())), keepdim=True).mul_(
                            centralization
                        )
                    )
                
                if weight_decay != 0:
                    # Perform weight decay
                    grad_weights = p_fp32.data.div(fim_base).div_(diff_fim_base)

                    rms = grad_weights.pow(2).mean().sqrt_()
                    divisor = max(clip, rms) / clip
                    grad_weights.div_(divisor)

                    p_fp32.data.add_(grad_weights, alpha=-lr*weight_decay)

                # Apply full step
                if group['cautious']:
                    # Apply caution as per 'Cautious Optimizers' - https://arxiv.org/abs/2411.16085
                    mask = (full_step * grad_nat > 0).to(full_step.dtype)
                    mask.div_(mask.mean().clamp_(min=1e-3))
                    full_step = full_step * mask
                p_fp32.data.add_(full_step, alpha=-lr)

                fim.mul_(fim_slow_beta).addcmul_(approx_grad_nat, approx_grad_nat, value=1 - fim_slow_beta).clamp_(-clip_lambda, clip_lambda)

                # pack
                if p.dtype in {torch.float16, torch.bfloat16}:
                    copy_stochastic_(state["fim"], fim)
                    copy_stochastic_(state["momentum"], momentum)
                    copy_stochastic_(state["prev_grad"], -grad)
                    copy_stochastic_(p, p_fp32)
                else:
                    # Copy the negative of the current grad (next step diff is -prev_grad + grad, or alternatively grad - prev_grad)
                    state["prev_grad"].copy_(-grad)
        return loss
