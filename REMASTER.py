# REMASTER from https://github.com/Clybius/Personalized-Optimizers by Clybius

import torch
from torch.optim import Optimizer
from math import sqrt

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

class REMASTER(Optimizer):
    r"""
    REMASTER: Applying the idea of no gradient accumulation, as its been superseded by momentum. Faster training, smoother weights, Papa Johns. 
    
    For optimal use: Utilize a gradient accumulation size of 1, highest batch size you can handle, adjust LR as needed. Standard AdamW LR ought to be stable enough.
    
    If you want extra speed, you can utilize the `reset_interval` and `reset_increment` parameter to reset the optimizer states, speeding up gradient descent and accelerating leaving local minima.

    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.0001).
        betas (float):
            Coefficient used for computing the running average, and the running square of running average (default: 0.95, 0.9999)
        weight_decay (float):
            AdamW-like weight decay, i.e. a L2 penalty (default: 0.0).
        weight_decay_rate (float):
            Decay the multiplier at which rate weight decay is applied, weight_decay * weight_decay_rate**step (default: 0.998).
        amp (float):
            Beta-adjusted scaling parameter for adding the running average to the gradient. (default: 5.0).
        reset_interval (int):
            Resets the optimizers running averages after (reset_interval + reset_increment * times_reset) steps (default: 0, recommended if used: >=100).
        reset_increment (int):
            Increments the reset_interval by this amount after every reset (default: 0, recommended if used: >=100).
        orthograd (bool):
            Modify the gradient to apply an orthogonal gradient update, - https://arxiv.org/abs/2501.04697 - extended with atan2 in place of epsilon - https://arxiv.org/abs/2407.05872 (default: False).
        cautious_min (bool):
            Use cautious mask on full step update, clamped to a minimum of cautious_min - https://arxiv.org/abs/2411.16085 (default: 1.0, thus disabling the mask. Use 0 to fully utilize the mask).
        stochastic_fp (bool):
            Utilize stochastic rounding for bf16 and fp16 tensors. (default: False).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple = (0.95, 0.9999),
        weight_decay: float = 0.0,
        weight_decay_rate: float = 0.998,
        amp: float = 5.0,
        reset_interval: int = 0,
        reset_increment: int = 0,
        orthograd: bool = True,
        cautious_min: float = 1.0,
        stochastic_fp: bool = True,
    ):

        self._init_lr = lr

        defaults = dict(
            lr = lr,
            betas = betas,
            weight_decay = weight_decay,
            weight_decay_rate = weight_decay_rate,
            amp = amp,
            reset_interval = reset_interval,
            reset_increment = reset_increment,
            orthograd = orthograd,
            cautious_min = cautious_min,
            stochastic_fp = stochastic_fp,
        )

        super(REMASTER, self).__init__(params, defaults)

    # Implementation from: https://github.com/LoganBooker/prodigy-plus-schedule-free/blob/1d2cfa2fe692a828d46a5a29b9667ec924961ac7/prodigyplus/core_optimiser.py#L169C5-L177C48
    @torch.no_grad()
    def orthograd(self, p):
        w = p.view(-1)
        g = p.grad.view(-1)

        proj = torch.dot(w, g).atan2_(torch.dot(w, w)).mul_(1.27323954474)
        g_orth = g.to(dtype=torch.float32, copy=True).sub_(w, alpha=proj)
        g_orth_scaled = g_orth.mul_(g.norm(2).div_(g_orth.norm(2).clamp_(min=1e-3)))

        p.grad.copy_(g_orth_scaled.view_as(p.grad))
    
    @torch.no_grad()
    def reset_momentums(self, momentum, sq_momentum):
        momentum.copy_(torch.zeros_like(momentum))
        sq_momentum.copy_(torch.zeros_like(sq_momentum))

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
            orthograd = group["orthograd"]
            step = group['step']

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]

                if orthograd and p.ndim >= 2:
                    self.orthograd(p)

                grad = p.grad.data

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["ema"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["ema_squared"] = torch.zeros_like(p.data)
                    # Optional resets
                    if group["reset_interval"] > 0:
                        state["times_zero"] = 0
                        state["steps_since_reset"] = 1

                p_fp32 = p.detach().clone()
                ema = state["ema"].detach().clone()
                ema_squared = state["ema_squared"].detach().clone()
                # Unpack
                if p.dtype in {torch.float16, torch.bfloat16} and group["stochastic_fp"]:
                    grad = grad.to(torch.float32)
                    ema = state['ema'].detach().clone().to(torch.float32)
                    ema_squared = state['ema_squared'].detach().clone().to(torch.float32)
                    p_fp32 = p.detach().clone().to(torch.float32)

                if group["reset_interval"] > 0:
                    if state["steps_since_reset"] // (group["reset_interval"] + (group["reset_increment"] * state["times_zero"])) > 0:
                        self.reset_momentums(ema, ema_squared)
                        state["times_zero"] += 1
                        state["steps_since_reset"] = 1
                    step = state["steps_since_reset"]

                slow_beta = ((betas[1]**step - betas[1]) / (betas[1]**step - 1.0))

                bias_correction = 1 - betas[0] ** step # Can apply to step_size, but this leads to significant initial updates and could be too much without warmup.
                bias_correction_sqrt = (1 - slow_beta ** step) ** (1 / 2)
                atan2_mul = 1.27323954474 # atan2(1,1) renormalization multiplier
                step_size = lr * atan2_mul

                # RMS Norm
                rms = grad.pow(2).mean().sqrt_().clamp_min_(1)
                grad.div_(rms)

                # Smooth EMA norm
                grad_norm, ema_norm = grad.norm(2), ema.norm(2)
                normalization_val = grad_norm.atan2(ema_norm).mul_(atan2_mul)

                if normalization_val > 1e-6:
                    grad.div_(normalization_val)
                
                # Update ema
                ema = ema.mul(betas[0]).add_(grad, alpha=1 - betas[0])

                # Adaptive ema
                mask = (grad * ema > 0).to(grad.dtype)
                mask.clamp_min_(betas[0])
                mask.div_(mask.mean().clamp_(min=1e-3)) # Divide by mean (0.001-1.0)
                ema = ema.mul(mask)

                # Compass amplification
                c_t = grad.add(ema.div(bias_correction), alpha=group["amp"])

                # AdamW debias
                denom = ema_squared.sqrt().div_(bias_correction_sqrt)

                # ADOPT update
                ema_squared = ema_squared.mul(slow_beta).addcmul_(c_t, c_t, value=1 - slow_beta)

                # Atan2-Adamw
                full_step = c_t.atan2(denom)

                if weight_decay != 0:
                    # Perform weight decay
                    grad_weights = p_fp32.data

                    full_step.add_(grad_weights, alpha=weight_decay * weight_decay_rate**group["step"])
                
                # Apply caution as per 'Cautious Optimizers' with a modified minimum.
                if group["cautious_min"] != 1.0:
                    mask = (full_step * grad > 0).to(full_step.dtype)
                    mask.clamp_min_(group["cautious_min"])
                    mask.div_(mask.mean().clamp_(min=1e-3))
                    full_step.mul_(mask)

                p_fp32.data.add_(full_step, alpha=-step_size)
                if p.dtype in {torch.float16, torch.bfloat16} and group["stochastic_fp"]:
                    copy_stochastic_(state["ema"], ema)
                    copy_stochastic_(state["ema_squared"], ema_squared)
                    copy_stochastic_(p, p_fp32)
                else:
                    state["ema"].copy_(ema)
                    state["ema_squared"].copy_(ema_squared)
                    p.copy_(p_fp32)
                if group["reset_interval"] > 0:
                    state["steps_since_reset"] += 1
        return loss