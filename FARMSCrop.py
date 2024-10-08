import torch
from torch.optim import Optimizer

class FARMSCrop(Optimizer):
    r"""
    FARMSCrop: Fisher-Accelerated RMSProp, replaced denom with momentum and compass-style amplification.
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.0001)
        betas (float, float):
            coefficients used for computing running averages of
            gradient difference FIM and approx. natural grad FIM (default: 0.999, 0.9999).
        eps (float):
            Term added to the denominator outside of the root operation to
            improve numerical stability. (default: 1e-8).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 1e-6).
        centralization (float):
            center model grad (default: 1.0).
        diff_mult (float):
            Multiplier for difference amplification (default: 1.0)
        momentum_beta (float):
            Beta value for slow momentum / EMA (default: 0.9999)
        momentum_amp (float):
            Amplification multiplier for slow momentum / EMA (default: 5.0)
    """

    def __init__(
        self,
        params,
        lr=1e-4,
        betas=(0.999, 0.9999),
        eps=1e-8,
        weight_decay=1e-6,
        centralization=1.0,
        diff_mult=1.0,
        momentum_beta=0.9999,
        momentum_amp=5.0,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            centralization=centralization,
            diff_mult=diff_mult,
            momentum_beta=momentum_beta,
            momentum_amp=momentum_amp,
        )
        super(FARMSCrop, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Fisher information matrix
                    state["fim"] = torch.ones_like(p.data)
                    # Fisher information matrix
                    state["momentum"] = torch.zeros_like(p.data)
                    # Prev grad
                    state["previous_grad"] = torch.zeros_like(p.data)
                    state["grad_diff_fim"] = torch.ones_like(p.data)

                fim = state["fim"]
                momentum = state["momentum"]

                beta1, beta2 = group["betas"]
                lr = group["lr"]
                weight_decay = group["weight_decay"]
                centralization = group["centralization"]
                diff_mult = group["diff_mult"]
                momentum_beta = group["momentum_beta"]
                momentum_amp = group["momentum_amp"]
                state["step"] += 1

                # bias correction step size
                bias_correction_sqrt = (1 - beta2 ** state["step"]) ** (1 / 2)
                fim_slow_beta = ((beta2**state["step"] - beta2) / (beta2**state["step"] - 1.0)) ** (1/2)
                step_size = lr

                # Get previous grad, initialized at 0 (first step is just grad)
                prev_grad = state["previous_grad"]
                # grad_diff will contain the difference between prev grad and current grad
                grad_diff = prev_grad.add(grad) * diff_mult

                grad_diff_fim = state["grad_diff_fim"]
                grad_diff_fim.mul_(beta1).addcmul_(grad_diff, grad_diff, value=1 - beta1)

                # Get natural gradient (squared ema, obtained sqrt of ema)
                diff_fim_base = grad_diff_fim.sqrt().add_(group["eps"])

                approx_grad_nat = grad.div(diff_fim_base)

                rms = approx_grad_nat.pow(2).mean().sqrt_()
                divisor = max(1, rms)
                approx_grad_nat.div_(divisor)

                fim.mul_(fim_slow_beta).addcmul_(approx_grad_nat, approx_grad_nat, value=1 - fim_slow_beta)
                fim_base = fim.sqrt().add_(group["eps"])

                grad_nat = grad.div(fim_base).mul_(diff_fim_base)
                rms = grad_nat.pow(2).mean().sqrt_()
                divisor = max(1, rms)
                grad_nat.div_(divisor)

                # center the gradient vector
                if centralization != 0 and grad_nat.dim() > 1:
                    grad_nat.sub_(
                        grad_nat.mean(dim=tuple(range(1, grad_nat.dim())), keepdim=True).mul_(
                            centralization
                        )
                    )

                # Compass-style amplification
                momentum.mul_(momentum_beta).add_(grad_nat, alpha=1 - momentum_beta)
                full_step = grad_nat.add(momentum, alpha=momentum_amp)

                if weight_decay != 0:
                    # Perform weight decay
                    grad_weights = p.data / fim_base * diff_fim_base

                    rms = grad_weights.pow(2).mean().sqrt_()
                    divisor = max(1, rms)
                    grad_weights.div_(divisor)

                    full_step.add_(grad_weights, alpha=weight_decay)

                # Apply full step
                p.data.add_(full_step, alpha=-step_size)
                # Copy the negative of the current grad (next step diff is -prev_grad + grad, or alternatively grad - prev_grad)
                state['previous_grad'].copy_(-grad)
        return loss
