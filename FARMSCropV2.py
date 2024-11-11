import torch
from torch.optim import Optimizer

class FARMSCropV2(Optimizer):
    r"""
    FARMSCropV2: Fisher-Accelerated RMSprop, with momentum-based Compass-style amplification, with ADOPT's AdamW changes. (https://arxiv.org/abs/2411.02853).
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
            Term added to the denominator outside of the root operation to
            improve numerical stability. (default: 1e-8).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 1e-6).
        centralization (float):
            Center model grad (default: 0.5).
        diff_mult (float):
            Multiplier for difference amplification (default: 0.25).
        momentum_beta (float):
            Beta value for slow momentum / EMA (default: 0.9999).
        momentum_amp (float):
            Amplification multiplier for slow momentum / EMA (default: 5.0).
        clip (float):
            Value to clip at (default: 1.0)
    """

    def __init__(
        self,
        params,
        lr=1e-4,
        betas=(0.999, 0.9999),
        eps=1e-8,
        weight_decay=1e-6,
        centralization=0.5,
        diff_mult=0.25,
        momentum_beta=0.9999,
        momentum_amp=5.0,
        clip=1.0,
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
            clip=clip,
        )
        super(FARMSCropV2, self).__init__(params, defaults)

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

                diff_mult = group["diff_mult"]
                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Fisher information matrix
                    state["fim"] = torch.ones_like(p.data)
                    # Fisher information matrix
                    state["momentum"] = torch.zeros_like(p.data)
                    # Prev grad
                    if diff_mult > 0:
                        state["previous_grad"] = torch.zeros_like(p.data)
                        state["grad_diff_fim"] = torch.ones_like(p.data)

                fim = state["fim"]
                momentum = state["momentum"]

                beta1, beta2 = group["betas"]
                lr = group["lr"]
                weight_decay = group["weight_decay"]
                centralization = group["centralization"]
                momentum_beta = group["momentum_beta"]
                momentum_amp = group["momentum_amp"]
                clip = group["clip"]
                state["step"] += 1

                fim_slow_beta = ((beta2**state["step"] - beta2) / (beta2**state["step"] - 1.0)) ** (1/2)

                # center the gradient vector
                if centralization != 0 and grad.dim() > 1:
                    grad.sub_(
                        grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True).mul_(
                            centralization
                        )
                    )

                if weight_decay != 0:
                    # Perform weight decay
                    grad_weights = p.data

                    if clip > 0:
                        rms = grad_weights.pow(2).mean().sqrt_()
                        divisor = max(clip, rms) / clip
                        grad_weights.div_(divisor)

                    p.data.add_(grad_weights, alpha=-weight_decay)

                if diff_mult > 0:
                    # Get previous grad, initialized at 0 (first step is just grad)
                    prev_grad = state["previous_grad"]
                    # grad_diff will contain the difference between prev grad and current grad
                    grad_diff = prev_grad.add(grad) * diff_mult

                    grad_diff_fim = state["grad_diff_fim"]

                    # Get natural gradient (squared ema, obtained sqrt of ema)
                    diff_fim_base = grad_diff_fim.sqrt().add_(group["eps"])
                    grad_diff_fim.mul_(beta1).addcmul_(grad_diff, grad_diff, value=1 - beta1)
                else:
                    diff_fim_base = 1.0

                approx_grad_nat = grad.div(diff_fim_base)

                fim_base = fim.sqrt().add_(group["eps"])

                grad_nat = grad.div(fim_base).mul_(diff_fim_base)

                # Compass-style amplification
                full_step = grad_nat.add(momentum, alpha=momentum_amp)

                if clip > 0:
                    rms = full_step.pow(2).mean().sqrt_()
                    divisor = max(clip, rms) / clip
                    full_step.div_(divisor)

                # Apply full step
                p.data.add_(full_step, alpha=-lr)

                fim.mul_(fim_slow_beta).addcmul_(approx_grad_nat, approx_grad_nat, value=1 - fim_slow_beta)

                momentum.mul_(momentum_beta).add_(grad_nat, alpha=1 - momentum_beta)

                if diff_mult > 0:
                    # Copy the negative of the current grad (next step diff is -prev_grad + grad, or alternatively grad - prev_grad)
                    state['previous_grad'].copy_(-grad)
        return loss
