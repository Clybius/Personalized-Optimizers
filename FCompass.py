import torch
from torch.optim import Optimizer

# Fisher optimizer (FAdam) from https://github.com/lessw2020/FAdam_PyTorch/blob/main/fadam.py by Less Wright (lessw2020), I may not know them, but I am aware of their expertise. Many thanks for your contributing work!
# Original optimizer (Compass) from https://github.com/lodestone-rock/compass_optimizer/blob/main/compass.py by lodestone-rock, many thanks for their optim, help, and ideas!
class FCompass(Optimizer):
    r"""
    Fisher Compass: Utilizing approximate fisher information to accelerate training. (Applied onto Compass).
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.001)
        betas (Tuple[float, float], optional):
            coefficients used for computing running averages of
            gradient and its square (default: (0.99, 0.999)).
        amp_fac (float):
            amplification factor for the first moment filter (default: 2).
        eps (float):
            Term added to the denominator outside of the root operation to
            improve numerical stability. (default: 1e-8).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0.1).
        clip (float):
            Clip gradient to this value (default: 1.0).
        centralization (float):
            Center grad (default: 1.0).
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.99, 0.999),
        amp_fac=2,
        eps=1e-8,
        weight_decay=0.1,
        clip=1.0,
        centralization=1.0,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            amp_fac=amp_fac,
            eps=eps,
            weight_decay=weight_decay,
            clip=clip,
            centralization=centralization,
        )
        super(FCompass, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Compass does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average and squared exponential moving average gradient values
                    state["momentum"] = torch.zeros_like(p.data)
                    state['max_ema_squared'] = torch.zeros_like(p.data)
                    # Fisher Information Matrix
                    state["fim"] = torch.ones_like(p.data)

                momentum, fim, max_ema_squared = state["momentum"], state["fim"], state['max_ema_squared']
                beta1, beta2 = group["betas"]
                amplification_factor = group["amp_fac"]
                lr = group["lr"]
                weight_decay = group["weight_decay"]
                clip = group["clip"]
                centralization = group["centralization"]
                state["step"] += 1

                # center the gradient vector
                if centralization != 0 and grad.dim() > 1:
                    grad.sub_(
                        grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True).mul_(centralization)
                    )

                # bias correction step size
                # soft warmup
                curr_beta2 = (beta2**state["step"] - beta2) / (beta2**state["step"] - 1.0)
                bias_correction_sqrt = (1 - curr_beta2 ** state["step"]) ** (1 / 2)

                # Update fim
                fim.mul_(curr_beta2).addcmul_(grad, grad, value=1 - curr_beta2)

                curr_eps = group["eps"] # Adaptive EPS births batman (NaN NaN NaN NaN... in grad when using mixed precision Stable Diffusion training) so we just use generic epsilon.

                fim_base = fim**0.5 + curr_eps

                # Compute natural gradient
                grad_nat = grad / fim_base

                if clip != 0:
                    rms = grad_nat.pow(2).mean().sqrt_().add_(curr_eps)
                    divisor = max(1, rms) / clip
                    grad_nat.div_(divisor)

                # Momentum + Compass amplification
                momentum.mul_(beta1).add_(grad_nat, alpha=1 - beta1)
                grad_nat.add_(momentum, alpha=amplification_factor)

                # Weight decay
                grad_weights = p.data / fim_base

                if clip != 0:
                    rms = grad_weights.pow(2).mean().sqrt_().add_(curr_eps)
                    divisor = max(1, rms) / clip
                    grad_weights.div_(divisor)
                
                full_step = grad_nat + (weight_decay * grad_weights)

                # Use the max. for normalizing running avg. of gradient (amsgrad)
                torch.max(max_ema_squared, max_ema_squared.mul(beta2).addcmul_(full_step, full_step, value=1 - beta2), out=max_ema_squared)
                denom = (max_ema_squared.sqrt() / bias_correction_sqrt).add_(curr_eps)

                p.data.addcdiv_(full_step, denom, value=-lr)
        return loss
