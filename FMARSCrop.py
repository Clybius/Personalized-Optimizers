import torch
from torch.optim import Optimizer

class FMARSCrop(Optimizer):
    r"""
    FMARSCrop: Fisher-accelerated MARS (https://arxiv.org/abs/2411.10438), with momentum-based Compass-style amplification, with ADOPT's AdamW changes (https://arxiv.org/abs/2411.02853).
    Un-official MARS implementation is credited to Less Wright (lessw2020).
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
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0.0).
        centralization (float):
            Center model grad (default: 0.0).
        moment_centralization (float):
            Center the slow momentum / EMA (default: 1.0).
        diff_mult (float):
            Multiplier for difference amplification (default: 1.0).
        momentum_beta (float):
            Beta value for slow momentum / EMA (default: 0.9999) (Alternative recommendation: 0.99999).
        momentum_lambda (float):
            Amplification exponent for slow momentum / EMA (default: 0.25) (Alternative recommendation: 0.5).
        clip (float):
            Value to clip the grad's RMS at (default: 1.0)
    """

    def __init__(
        self,
        params,
        lr=1e-4,
        betas=(0.999, 0.9999),
        eps=1e-6,
        weight_decay=0.0,
        centralization=0.0,
        moment_centralization=1.0,
        diff_mult=1.0,
        momentum_beta=0.9999,
        momentum_lambda=0.25,
        clip=1.0,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            centralization=centralization,
            moment_centralization=moment_centralization,
            diff_mult=diff_mult,
            momentum_beta=momentum_beta,
            momentum_lambda=momentum_lambda,
            clip=clip,
        )
        super(FMARSCrop, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            # Get parameters with gradients
            params_with_grad = []
            grads = []
            momentums = []
            fims = []
            prev_grads = []
            grad_diff_fims = []
            state_steps = []

            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            centralization = group["centralization"]
            moment_centralization = group["moment_centralization"]
            diff_mult = group["diff_mult"]
            momentum_beta = group["momentum_beta"]
            momentum_lambda = group["momentum_lambda"]
            clip = group["clip"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["momentum"] = torch.zeros_like(p)
                    state["fim"] = torch.ones_like(p)
                    state["prev_grad"] = -p.grad.clone().detach()
                    if diff_mult > 0:
                        state["grad_diff_fim"] = torch.ones_like(p)

                momentums.append(state["momentum"])
                fims.append(state["fim"])
                prev_grads.append(state["prev_grad"])
                if diff_mult > 0:
                    grad_diff_fims.append(state["grad_diff_fim"])
                state["step"] += 1
                state_steps.append(state["step"])

            for i, p in enumerate(params_with_grad):
                grad = grads[i]
                prev_grad = prev_grads[i].add(grad)

                step = state_steps[i]

                # Calculate cₜ (gradient with correction term)
                correction = (1 - beta1) / 2 * beta1 / (1 - beta1) * prev_grad
                c_t = grad + correction

                # Gradient clipping (if necessary)
                grad_norm = torch.norm(c_t)
                if grad_norm > clip:
                    c_t = c_t * clip / grad_norm

                fim = fims[i]
                momentum = momentums[i]
                if moment_centralization != 0:
                    momentum = momentum - torch.mean(momentum)

                clip_lambda = step**0.25

                fim_slow_beta = ((beta2**step - beta2) / (beta2**step - 1.0)) ** (1/2)

                approx_grad_nat = c_t

                if diff_mult > 0:
                    # Get previous grad, initialized at 0 (first step is just grad)
                    #prev_grad = prev_grads[i]
                    # grad_diff will contain the difference between prev grad and current grad
                    grad_diff = prev_grad * diff_mult

                    rms = grad_diff.pow(2).mean().sqrt_()
                    divisor = max(clip, rms) / clip
                    grad_diff.div_(divisor)

                    grad_diff_fim = grad_diff_fims[i]

                    # Get natural gradient (squared ema, obtained sqrt of ema)
                    diff_fim_base = torch.clamp(grad_diff_fim.sqrt(), group["eps"])

                    grad_diff_fims[i].mul_(beta1).addcmul_(grad_diff, grad_diff, value=1 - beta1).clamp_(-clip_lambda, clip_lambda)
                else:
                    diff_fim_base = 1.0

                approx_grad_nat.div_(diff_fim_base)
                rms = approx_grad_nat.pow(2).mean().sqrt_()
                divisor = max(clip, rms) / clip
                approx_grad_nat.div_(divisor)

                fim_base = torch.clamp(fim.sqrt(), group["eps"])

                grad_nat = c_t.div(fim_base).div_(diff_fim_base)
                rms = grad_nat.pow(2).mean().sqrt_()
                divisor = max(clip, rms) / clip
                grad_nat.div_(divisor)

                # Compass-style amplification
                full_step = grad_nat.add(momentum, alpha=step**momentum_lambda)

                # center the gradient vector
                if centralization != 0 and full_step.dim() > 1:
                    full_step.sub_(
                        full_step.mean(dim=tuple(range(1, full_step.dim())), keepdim=True).mul_(
                            centralization
                        )
                    )
                
                if weight_decay != 0:
                    # Perform weight decay
                    grad_weights = p.data.div(fim_base).div_(diff_fim_base)

                    rms = grad_weights.pow(2).mean().sqrt_()
                    divisor = max(clip, rms) / clip
                    grad_weights.div_(divisor)

                    p.data.add_(grad_weights, alpha=-lr*weight_decay)

                # Apply full step
                p.data.add_(full_step, alpha=-lr)

                fims[i].mul_(fim_slow_beta).addcmul_(approx_grad_nat, approx_grad_nat, value=1 - fim_slow_beta).clamp_(-clip_lambda, clip_lambda)

                momentums[i].mul_(momentum_beta).add_(grad_nat, alpha=1 - momentum_beta)

                # Copy the negative of the current grad (next step diff is -prev_grad + grad, or alternatively grad - prev_grad)
                prev_grads[i].copy_(-grad)
        return loss