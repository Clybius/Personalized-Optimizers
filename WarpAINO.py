"""WarpAINO: Adaptive Innovation-Normalized Orthogonalized optimizer, a gradient
optimizer that builds a signed crafted update and adapts it with a learned warp.

Each optimizer step processes a gradient through nine stages. Stages 1 to 5 craft
the base update; stage 6 applies a learned linear warp and an orthogonalization;
stages 7 to 9 normalize, mask, and apply weight decay.

Stage 1 (normalize). The gradient is rescaled so its energy is balanced. For 2D
and higher-rank parameters this is an iterative Sinkhorn normalization that
alternates row and column RMS balancing ``sinkhorn_steps`` times. For 1D and
scalar parameters it is a single RMS normalization.

Stage 2 (sign momentum). A running sign statistic of the normalized gradient is
updated with the coefficient ``beta1``. The statistic tracks which
directions the gradient consistently points, averaged over time.

Stage 3 (squared innovation). The change from the previous normalized value is
squared and tracked with the Polyak-corrected coefficient ``beta2``. For 2D
parameters the square is factorized into a row factor and a column factor; for
1D parameters it is elementwise. This factorized denominator downweights axes
that change little.

Stage 4 (value momentum). The normalized gradient is blended into a running
value statistic with the Polyak-corrected coefficient ``beta1``.

Stage 5 (craft). The crafted update is the absolute value of the value momentum
divided by the innovation denominator, signed by the sign momentum.

Stage 6 (warp). The crafted update is linearly modified by a learned operator
that starts at the identity. In dense mode the warp is a residual matrix D so
update_warped = update + D @ update. In spectral mode bounded positive
log-frequency gains are applied through orthonormal FFTs. 2D parameters are then
orthogonalized by a two-step Gram-Newton-Schulz iteration. For the first step
(step_t <= 1) the warp is bypassed so training begins as the unwarped craft.

Stage 7 (variance track). A running variance of the warped update is updated with
the Polyak-corrected coefficient ``beta3``.

Stage 8 (normalize and rescale). The warped update is divided by its tracked
variance and multiplied back to the RMS norm of the warped update, so the update
scale is preserved across steps.

Stage 9 (cautious mask and weight decay). Components of the update that point
opposite to the gradient sign are zeroed and the survivor mean is restored. A
decoupled weight decay then subtracts a fraction of the parameter; with
``relative_wd`` the fraction is balanced by the update-to-parameter RMS ratio,
and with ``cautious_wd`` decay is skipped where it would flip the parameter sign.

The warp learns in one SGD step per optimizer step (no closure) by predicting
the next crafted update. The prediction loss is the squared norm of
(P @ u_{t-1} - u_t) divided by the squared norm of u_{t-1}, where P is the warp
operator (I + D in dense mode). ``meta_wd`` damps the warp toward zero each step
so it does not overfit. Setting ``meta_lr=0`` disables the warp entirely.

------------------------------------------------------------------------------
PARAMETERS (passed to ``WarpAINO(params, ...)``)
------------------------------------------------------------------------------
Core optimization:
  lr (float, default 0.0001)
      Master learning rate. Scales how far each parameter moves each step
      (applied as ``param -= lr * update``).

  betas (tuple of 3 floats, default (0.95, 0.95, 0.999))
      Smoothing coefficients for the three tracked statistics:
        - beta1 (0..1): momentum for the tracked value and for the
          gradient-sign momentum.
        - beta2 (0..1): momentum for the squared-innovation denominator,
          which is factorized into a row and a column term for 2D+ params
          and elementwise for 1D params.
        - beta3 (0..1): momentum for the running variance used to
          normalize and rescale the final update.

  weight_decay (float, default 0.0)
      Decoupled weight decay strength. Subtracts a fraction of the parameter
      itself each step to regularize. Set to 0 to disable.

  eps (float, default 1e-16)
      Small numerical stabilizer added inside divisions/norms to avoid
      division by zero. Usually left at the default.

Cautious-gradient behavior (skip updates that disagree with the gradient):
  cautious_update (bool, default True)
      If True, zero out (and renormalize) any part of the update that points
      opposite to the current gradient sign. This prevents harmful steps.

  cautious_wd (bool, default True)
      If True, only apply weight decay where the update and the parameter
      share the same sign (i.e. decay does not flip the parameter's sign).

  relative_wd (bool, default True)
      If True, scale the weight-decay strength relative to the update's RMS
      vs the parameter's RMS, so decay is balanced across parameters of
      different magnitudes. If False, decay is applied at the raw
      ``weight_decay`` rate.

Crafted-update construction:
  sinkhorn_steps (int, default 5)
      Number of Sinkhorn normalization iterations applied to 2D+ gradients to
      balance row/column energies before crafting the update. Higher = more
      balanced (and slightly slower).

  ortho_dtype (torch.dtype, default torch.bfloat16)
      Low-precision dtype used inside the Gram-Newton-Schulz orthogonalization
      of the warped update (2D+ params only). Reduces memory/compute; results
      are upcast back to the parameter dtype.

Performance / dtype options:
  stochastic_fp (bool, default True)
      If True (and params are bfloat16), stochastically round the updated
      values back into the param instead of rounding deterministically, which
      can improve low-precision training.

  compile_step (bool, default False)
      If True, JIT-compile (torch.compile) the per-parameter core steps for
      speed. Falls back gracefully if compilation fails.

  foreach (bool, default False)
      If True, batch the final update across all 2D+ parameters in a group
      using fused ``foreach`` kernels for extra throughput.

Warp (the learned adaptation):
  warp_mode (str, default "spectral", choices "dense"/"spectral")
      Which warp to build:
        - "dense": a dense residual matrix D with update_warped = update + D @ update.
        - "spectral": bounded positive log-frequency scales applied through
          orthonormal FFTs (default, lower memory).

  meta_lr (float, default 0.05)
      Learning rate for the warp's own update (predicting the next crafted
      update). Set to 0 to disable the warp; the optimizer then applies the
      unwarped craft.

  meta_wd (float, default 0.01)
      Weight decay / damping on the warp itself, pulling it back toward zero so
      the optimizer does not overfit its warp.

  warp_dtype (torch.dtype, default torch.bfloat16)
      Storage dtype for the dense warp residual D (dense mode). FP32 keeps the
      warp most precise; bfloat16/float16 save memory.

  spectral_bilateral (bool, default True)
      (Spectral mode, 2D+ params only.) If True, apply independent warp
      operators on both the row and column dimensions; if False, only rows.

  spectral_log_bound (float, default 1.0)
      Maximum absolute log-gain of the spectral warp (clamps how strong a
      single frequency can scale). Larger allows stronger warps.
"""

import torch
from torch.optim import Optimizer
from typing import Tuple, List, Union, Iterable, Optional

GRAM_NEWTON_SCHULZ_2STEP_COEFFS1 = [
    (1.4897216394163149, -0.5798724169434551, 0.0831346315615072),
    (2.0181598271548000, -1.5523232773433393, 0.5343894201774000),
]


# ---------------------------------------------------------------------------
# Low-level shared helpers
# ---------------------------------------------------------------------------

def copy_stochastic_(target: torch.Tensor, source: torch.Tensor) -> None:
    """Stochastically round a float32 source into bfloat16."""
    assert source.dtype is torch.float32, f"source must be float32, got {source.dtype}"
    assert target.dtype is torch.bfloat16, f"target must be bfloat16, got {target.dtype}"
    with torch.no_grad():
        result = torch.randint_like(source, dtype=torch.int32, low=0, high=(1 << 16))
        result.add_(source.view(dtype=torch.int32))
        result.bitwise_and_(-65536)
        target.copy_(result.view(dtype=torch.float32).to(target.dtype))


def _reshape_to_2d(t: torch.Tensor) -> torch.Tensor:
    """Reshape tensor to 2D: [N, -1] for >2D, [1, -1] for 1D, identity for 2D."""
    if t.ndim > 2:
        return t.reshape(len(t), -1)
    if t.ndim < 2:
        return t.reshape(1, -1)
    return t


def _spectral_mirror(x: torch.Tensor) -> torch.Tensor:
    """Reflect a full FFT spectrum onto its negative-frequency indices."""
    return torch.roll(x.flip(0), shifts=1, dims=0)


def _spectral_log_gain(
    log_scale: torch.Tensor,
    log_bound: float,
) -> torch.Tensor:
    """Return real conjugate-symmetric positive gains for a FFT warp."""
    log_scale = 0.5 * (log_scale + _spectral_mirror(log_scale))
    return log_scale.clamp(-log_bound, log_bound).exp()


def _spectral_apply(
    update_2d: torch.Tensor,
    spectral_log_left: torch.Tensor,
    spectral_log_right: Optional[torch.Tensor],
    spectral_log_bound: float,
) -> torch.Tensor:
    """Apply a full-rank symmetric-circulant warp using orthonormal FFTs."""
    left_gain = _spectral_log_gain(spectral_log_left, spectral_log_bound)
    if spectral_log_right is None:
        spectrum = torch.fft.fft(update_2d, dim=0, norm="ortho")
        delta = torch.fft.ifft(
            spectrum * (left_gain[:, None] - 1.0), dim=0, norm="ortho"
        ).real
    else:
        right_gain = _spectral_log_gain(spectral_log_right, spectral_log_bound)
        spectrum = torch.fft.fft2(update_2d, dim=(0, 1), norm="ortho")
        gain_delta = left_gain[:, None] * right_gain[None, :] - 1.0
        delta = torch.fft.ifft2(
            spectrum * gain_delta, dim=(0, 1), norm="ortho"
        ).real
    return update_2d + delta


@torch.no_grad()
def gram_newton_schulz_2step(
    M: torch.Tensor,
    eps: float = 1e-7,
    ortho_dtype=torch.bfloat16,
) -> torch.Tensor:
    """2-step Gram Newton-Schulz with pre-optimized unconstrained coefficients."""
    X = M.to(ortho_dtype)
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True

    # AOL-Gram folding
    A = X @ X.mT
    rescaling = A.abs().sum(dim=-1).clamp_min_(eps)
    s = rescaling.rsqrt().unsqueeze(-1)
    X = X * s
    R = s * A * s.mT

    n, m = X.shape
    I = torch.eye(n, dtype=X.dtype, device=X.device)
    Q = I

    # Apply pre-optimized coefficients
    for a, b, c in GRAM_NEWTON_SCHULZ_2STEP_COEFFS1:
        R2 = R @ R
        z = a * I + b * R + c * R2
        Q = Q @ z
        R = z @ R @ z

    out = Q @ X

    if transposed:
        out = out.T

    return out.to(M.dtype)


@torch.no_grad()
def _warp_meta_update(
    state: dict,
    update_cur_2d: torch.Tensor,
    meta_lr: float,
    meta_wd: float,
) -> None:
    """Update the dense residual on the normalized prediction objective.

    The stored residual is D in P = I + D. It is kept in low precision to
    reduce optimizer-state memory, but the update is accumulated in FP32
    before being quantized back to the configured storage dtype.
    """
    warp = state["warp"]
    prev = state.get("prev_update")
    if prev is not None:
        warp_fp32 = warp.float()
        R = (prev + warp_fp32 @ prev) - update_cur_2d
        scale = 1.0 / prev.norm().square().clamp_min_(1e-12)
        # Fuse the prediction-residual matmul and accumulation into a single
        # asynchronous addmm_ (no intermediate (m,m) tensor materialized).
        warp_fp32.addmm_(R, prev.t(), beta=1.0, alpha=-meta_lr * scale)
        if meta_wd > 0:
            warp_fp32.mul_(1.0 - meta_lr * meta_wd)
        warp.copy_(warp_fp32.to(warp.dtype))
    # crafted_update is a fresh tensor produced by the core each step, so the
    # stored previous update does not need to be cloned.
    state["prev_update"] = update_cur_2d


@torch.no_grad()
def _spectral_meta_update(
    state: dict,
    update_cur_2d: torch.Tensor,
    meta_lr: float,
    meta_wd: float,
    spectral_log_bound: float,
) -> None:
    """Update FFT log-scales on the normalized prediction objective."""
    prev = state.get("prev_update")
    if prev is not None:
        log_left = state["spectral_log_left"]
        left_gain = _spectral_log_gain(log_left, spectral_log_bound)
        scale = 1.0 / prev.norm().square().clamp_min_(1e-12)

        log_right = state.get("spectral_log_right")
        if log_right is None:
            prev_spectrum = torch.fft.fft(prev, dim=0, norm="ortho")
            current_spectrum = torch.fft.fft(update_cur_2d, dim=0, norm="ortho")
            prediction = left_gain[:, None] * prev_spectrum
            residual = prediction - current_spectrum
            grad_left = left_gain * torch.real(
                torch.conj(residual) * prev_spectrum
            ).sum(dim=1)
            grad_left.mul_(scale)
        else:
            right_gain = _spectral_log_gain(log_right, spectral_log_bound)
            prev_spectrum = torch.fft.fft2(prev, dim=(0, 1), norm="ortho")
            current_spectrum = torch.fft.fft2(
                update_cur_2d, dim=(0, 1), norm="ortho"
            )
            base_left = prev_spectrum * right_gain[None, :]
            prediction = left_gain[:, None] * base_left
            residual = prediction - current_spectrum
            grad_left = left_gain * torch.real(
                torch.conj(residual) * base_left
            ).sum(dim=1)
            base_right = prev_spectrum * left_gain[:, None]
            grad_right = right_gain * torch.real(
                torch.conj(residual) * base_right
            ).sum(dim=0)
            grad_left.mul_(scale)
            grad_right.mul_(scale)

        log_left.add_(grad_left, alpha=-meta_lr)
        if meta_wd > 0:
            log_left.mul_(1.0 - meta_lr * meta_wd)
        log_left.clamp_(-spectral_log_bound, spectral_log_bound)
        log_left.copy_(0.5 * (log_left + _spectral_mirror(log_left)))

        if log_right is not None:
            log_right.add_(grad_right, alpha=-meta_lr)
            if meta_wd > 0:
                log_right.mul_(1.0 - meta_lr * meta_wd)
            log_right.clamp_(-spectral_log_bound, spectral_log_bound)
            log_right.copy_(0.5 * (log_right + _spectral_mirror(log_right)))

    state["prev_update"] = update_cur_2d.clone()


# ---------------------------------------------------------------------------
# De-duplicated math helpers (each originally repeated across the core fns)
# ---------------------------------------------------------------------------

def _poly_beta(beta: float, step_t: torch.Tensor) -> torch.Tensor:
    """Polyak-style beta correction used by the momentum, innovation, and variance updates."""
    beta_pow = beta ** step_t
    return torch.where(
        step_t > 1.0,
        (beta_pow - beta) / (beta_pow - 1.0),
        torch.zeros_like(beta_pow),
    )


def _cautious(update_final: torch.Tensor, g: torch.Tensor, cautious_update: bool) -> torch.Tensor:
    """Cautious masking: zero out the update where it disagrees in sign with g."""
    if cautious_update:
        mask = (g * update_final > 0).to(update_final.dtype)
        mask_mean = mask.mean().clamp_min(1e-3)
        update_final = update_final * mask / mask_mean
    return update_final


def apply_weight_decay(
    p: torch.Tensor,
    update_final: torch.Tensor,
    g: torch.Tensor,
    weight_decay: float,
    eps: float,
    cautious_update: bool,
    cautious_wd: bool,
    relative_wd: bool,
    target_rms: torch.Tensor,
) -> torch.Tensor:
    """Decoupled weight decay (absolute or RMS-relative), shared by all cores.

    ``target_rms`` is the RMS of the (un-cautious-masked) crafted/warp update,
    which the caller precomputes; when ``cautious_update`` is False we reuse it
    to avoid a redundant full-tensor reduction (matching the original cores).
    """
    if weight_decay == 0:
        return update_final

    if relative_wd:
        # Reuse target_rms when no cautious masking would have changed the
        # update's RMS (avoids a redundant full-tensor reduction).
        if cautious_update:
            update_rms = torch.sqrt(update_final.pow(2).mean() + eps)
        else:
            update_rms = target_rms
        param_rms = torch.sqrt(p.pow(2).mean() + eps)
        wd_coef = weight_decay * (update_rms / param_rms.clamp_min(eps))
        wd_full = wd_coef * p
        # Overshoot protection (independent of cautious_wd): if the full
        # wd term would reverse this update element's sign, zero it.
        flip = (update_final * (update_final + wd_full)) < 0
        if cautious_wd:
            wd_mask = (update_final.sign() == p.sign()).to(update_final.dtype)
            wd_term = wd_full * wd_mask
        else:
            wd_term = wd_full
        return (update_final + wd_term) * (~flip)
    elif cautious_wd:
        wd_mask = (update_final.sign() == p.sign()).to(update_final.dtype)
        return update_final + weight_decay * p * wd_mask
    else:
        return update_final + weight_decay * p


def _normalize_and_rescale(
    O: torch.Tensor,
    row_var: torch.Tensor,
    poly_beta3: torch.Tensor,
    eps: float,
    ref: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Track variance, normalize, and rescale ``O`` back to ``ref``'s RMS norm.

    Returns ``(update_final, target_rms)``. For 2D params ``row_var`` is
    per-row; for 1D/0D it is scalar -- the only structural difference, handled
    by branching on the tensor dimensionality.
    """
    target_rms = torch.sqrt(ref.pow(2).mean() + eps)
    if O.dim() >= 2:
        row_sq = O.pow(2).mean(dim=-1)
        row_var.lerp_(row_sq, 1.0 - poly_beta3)
        O_norm = O / torch.sqrt(row_var.clamp_min(eps).unsqueeze(-1))
    else:
        var_sq = O.pow(2).mean()
        row_var.lerp_(var_sq, 1.0 - poly_beta3)
        O_norm = O / torch.sqrt(row_var.clamp_min(eps))
    current_rms = torch.sqrt(O_norm.pow(2).mean() + eps)
    update_final = O_norm * (target_rms / (current_rms + eps))
    return update_final, target_rms


def _aino_craft_2d(
    g_2d: torch.Tensor,
    momentum: torch.Tensor,
    sign_momentum: torch.Tensor,
    exp_avg_sq_row: torch.Tensor,
    exp_avg_sq_col: torch.Tensor,
    beta1: float,
    beta2: float,
    beta3: float,
    step_t: torch.Tensor,
    sinkhorn_steps: int,
    eps: float,
) -> torch.Tensor:
    """Shared crafted-update stages (1 to 5) for 2D+ params.

    Sinkhorn normalization, gradient sign momentum, factorized row and column
    squared innovation, value momentum, and the signed craft. State tensors are
    mutated in place; returns the crafted ``update``.
    """
    # 1. Sinkhorn normalization for 2D+ parameters
    g_norm = g_2d
    for _ in range(sinkhorn_steps):
        row_rms = torch.sqrt(g_norm.pow(2).mean(dim=-1, keepdim=True) + eps)
        g_norm = g_norm / row_rms
        col_rms = torch.sqrt(g_norm.pow(2).mean(dim=-2, keepdim=True) + eps)
        g_norm = g_norm / col_rms

    poly_beta1 = _poly_beta(beta1, step_t)
    poly_beta2 = _poly_beta(beta2, step_t)

    # 2. Track gradient sign momentum using poly_beta1
    g_sign = g_norm.sign()
    sign_momentum.lerp_(g_sign, 1.0 - beta1)

    # 3. Track factorized row & column squared innovation using poly_beta2
    diff_sq = (g_norm - momentum).pow(2)
    exp_avg_sq_row.lerp_(diff_sq.mean(dim=-1, keepdim=True), 1.0 - poly_beta2)
    exp_avg_sq_col.lerp_(diff_sq.mean(dim=-2, keepdim=True), 1.0 - poly_beta2)

    # Construct factorized denominator
    r_factor = (exp_avg_sq_row + eps).sqrt()
    c_factor = ((exp_avg_sq_col + eps) / (exp_avg_sq_col.mean(dim=-1, keepdim=True) + eps)).sqrt()
    denom = r_factor * c_factor

    # 4. Update tracked value momentum using poly_beta1
    momentum.lerp_(g_norm, 1.0 - poly_beta1)

    # 5. Craft update: apply tracked sign to absolute value of (momentum / denom)
    return sign_momentum * (momentum.abs() / denom)


def _aino_craft_1d(
    g_data: torch.Tensor,
    momentum: torch.Tensor,
    sign_momentum: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    beta1: float,
    beta2: float,
    beta3: float,
    step_t: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Shared crafted-update stages (1 to 5) for 1D/0D params.

    RMS normalization, gradient sign momentum, squared innovation, value
    momentum, and the signed craft. State tensors are mutated in place;
    returns the crafted ``update``. (Orthogonalization is skipped for 1D.)
    """
    # 1. RMS normalize gradient for 1D/0D parameters
    grad_rms = torch.sqrt(g_data.pow(2).mean() + eps)
    g_norm = g_data / grad_rms

    poly_beta1 = _poly_beta(beta1, step_t)
    poly_beta2 = _poly_beta(beta2, step_t)

    # 2. Track gradient sign momentum using poly_beta1
    g_sign = g_norm.sign()
    sign_momentum.lerp_(g_sign, 1.0 - beta1)

    # 3. Track squared innovation relative to un-updated momentum using poly_beta2
    diff_sq = (g_norm - momentum).pow(2)
    exp_avg_sq.lerp_(diff_sq, 1.0 - poly_beta2)
    denom = exp_avg_sq.sqrt().clamp_min_(eps)

    # 4. Update tracked value momentum using poly_beta1
    momentum.lerp_(g_norm, 1.0 - poly_beta1)

    # 5. Craft update: apply tracked sign to absolute value of (momentum / denom)
    return sign_momentum * (momentum.abs() / denom)


# ---------------------------------------------------------------------------
# Unified cores
# ---------------------------------------------------------------------------

class WarpAINO(Optimizer):
    """Gradient optimizer that builds a signed crafted update and adapts it with a learned linear warp.

    The normalized gradient first passes through the craft stages: sign
    momentum, factorized squared innovation, value momentum, and a signed
    craft. The crafted update is then linearly warped by an identity-anchored
    operator immediately before orthogonalization. The default dense mode uses:

        update_warped = update + D @ update,   D = 0

    Spectral mode uses bounded positive log-frequency scales and orthonormal
    FFTs. In 2D+ bilateral mode it applies independent symmetric-circulant
    operators on the row and column dimensions.

    D is zero-initialized so P = I exactly and the warp has no effect until the
    residual learns. After each step, D is updated with one SGD step (no
    closure required) on the one-step-ahead prediction loss of the crafted
    update: ||P @ u_{t-1} - u_t||^2 / ||u_{t-1}||^2; meta_wd damps D back toward
    0. meta_lr=0 disables the warp and the optimizer applies the unwarped craft.

    The dense residual is stored in ``warp_dtype`` and promoted to FP32 for
    matrix multiplication and meta-updates.
    """

    def __init__(
        self,
        params,
        lr: float = 0.0001,
        betas: Tuple[float, float, float] = (0.95, 0.95, 0.999),
        weight_decay: float = 0.0,
        cautious_update: bool = True,
        cautious_wd: bool = True,
        stochastic_fp: bool = True,
        compile_step: bool = False,
        foreach: bool = False,
        sinkhorn_steps: int = 5,
        ortho_dtype: torch.dtype = torch.bfloat16,
        eps: float = 1e-16,
        meta_lr: float = 5e-2,
        meta_wd: float = 1e-2,
        warp_dtype: torch.dtype = torch.bfloat16,
        warp_mode: str = "spectral",
        spectral_bilateral: bool = True,
        spectral_log_bound: float = 1.0,
        relative_wd: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if len(betas) != 3:
            raise ValueError(f"betas must be a 3-tuple of floats, got {betas}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0 (beta1): {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1 (beta2): {betas[1]}")
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 2 (beta3): {betas[2]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps value: {eps}")
        if meta_lr < 0.0:
            raise ValueError(f"Invalid meta_lr value: {meta_lr}")
        if meta_wd < 0.0:
            raise ValueError(f"Invalid meta_wd value: {meta_wd}")
        if warp_mode not in ("dense", "spectral"):
            raise ValueError(
                f"Invalid warp_mode: {warp_mode}; expected 'dense' or 'spectral'"
            )
        if spectral_log_bound < 0.0:
            raise ValueError(
                f"Invalid spectral_log_bound value: {spectral_log_bound}"
            )
        if warp_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(
                "warp_dtype must be torch.float16, torch.bfloat16, or torch.float32, "
                f"got {warp_dtype}"
            )
        if not isinstance(relative_wd, bool):
            raise ValueError(f"relative_wd must be bool, got {relative_wd}")

        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            cautious_update=cautious_update,
            cautious_wd=cautious_wd,
            stochastic_fp=stochastic_fp,
            sinkhorn_steps=sinkhorn_steps,
            ortho_dtype=ortho_dtype,
            eps=eps,
            meta_lr=meta_lr,
            meta_wd=meta_wd,
            warp_dtype=warp_dtype,
            warp_mode=warp_mode,
            spectral_bilateral=spectral_bilateral,
            spectral_log_bound=spectral_log_bound,
            relative_wd=relative_wd,
        )
        super().__init__(params, defaults)

        self._compile_step = compile_step
        self._foreach = foreach

        if self._compile_step:
            import logging

            torch._dynamo.config.recompile_limit = max(
                torch._dynamo.config.recompile_limit, 64
            )
            try:
                self._compiled_2d = torch.compile(
                    self._aino_core_2d,
                    fullgraph=True,
                    dynamic=False,
                )
                self._compiled_1d = torch.compile(
                    self._aino_core_1d,
                    fullgraph=True,
                    dynamic=False,
                )
            except Exception as e:
                logging.warning(
                    f"torch.compile failed for WarpAINO cores: {e}. "
                    "Falling back to uncompiled cores."
                )
                self._compiled_2d = self._aino_core_2d
                self._compiled_1d = self._aino_core_1d
        else:
            self._compiled_2d = self._aino_core_2d
            self._compiled_1d = self._aino_core_1d

    @staticmethod
    def _aino_core_2d(
        p_2d: torch.Tensor,
        g_2d: torch.Tensor,
        momentum: torch.Tensor,
        sign_momentum: torch.Tensor,
        exp_avg_sq_row: torch.Tensor,
        exp_avg_sq_col: torch.Tensor,
        row_var: torch.Tensor,
        beta1: float,
        beta2: float,
        beta3: float,
        weight_decay: float,
        eps: float,
        step_t: torch.Tensor,
        sinkhorn_steps: int,
        cautious_update: bool,
        cautious_wd: bool,
        ortho_dtype: torch.dtype,
        relative_wd: bool,
        warp: Optional[torch.Tensor],
        spectral_log_left: Optional[torch.Tensor],
        spectral_log_right: Optional[torch.Tensor],
        spectral_bilateral: bool,
        spectral_log_bound: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unified 2D+ step: craft update, optionally warp, then process.

        Dispatch is driven by the state present (spectral / dense / none).
        Returns ``(update_final, crafted_update)`` for the meta-update.
        """
        # 1-5. Shared crafted-update stages.
        update = _aino_craft_2d(
            g_2d, momentum, sign_momentum, exp_avg_sq_row, exp_avg_sq_col,
            beta1, beta2, beta3, step_t, sinkhorn_steps, eps,
        )
        poly_beta3 = _poly_beta(beta3, step_t)

        # 6. Warp the crafted update (if any), then orthogonalize.
        if spectral_log_left is not None:
            if spectral_bilateral and spectral_log_right is not None:
                warped = _spectral_apply(update, spectral_log_left, spectral_log_right, spectral_log_bound)
            else:
                warped = _spectral_apply(update, spectral_log_left, None, spectral_log_bound)
            update_warped = torch.where(step_t <= 1.0, update, warped)
            O = gram_newton_schulz_2step(update_warped, eps=1e-7, ortho_dtype=ortho_dtype)
        elif warp is not None:
            update_warped = update + warp.float() @ update
            O = gram_newton_schulz_2step(update_warped, eps=1e-7, ortho_dtype=ortho_dtype)
        else:
            update_warped = update
            O = gram_newton_schulz_2step(update, eps=1e-7, ortho_dtype=ortho_dtype)

        # 7-8. Variance-normalize + rescale back to the warped update's RMS.
        update_final, target_rms = _normalize_and_rescale(O, row_var, poly_beta3, eps, update_warped)

        # 9. Cautious + decoupled weight decay.
        update_final = _cautious(update_final, g_2d, cautious_update)
        update_final = apply_weight_decay(
            p_2d, update_final, g_2d, weight_decay, eps,
            cautious_update, cautious_wd, relative_wd, target_rms,
        )

        return update_final, update

    @staticmethod
    def _aino_core_1d(
        p_data: torch.Tensor,
        g_data: torch.Tensor,
        momentum: torch.Tensor,
        sign_momentum: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        row_var: torch.Tensor,
        beta1: float,
        beta2: float,
        beta3: float,
        weight_decay: float,
        eps: float,
        step_t: torch.Tensor,
        cautious_update: bool,
        cautious_wd: bool,
        relative_wd: bool,
        warp: Optional[torch.Tensor],
        spectral_log_left: Optional[torch.Tensor],
        spectral_log_bound: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unified 1D/0D step: craft update, optionally warp, then process.

        Dispatch is driven by the state present (spectral / dense / none).
        Orthogonalization is skipped for 1D. Returns
        ``(update_final, crafted_update)`` for the meta-update.
        """
        # 1-5. Shared crafted-update stages.
        update = _aino_craft_1d(
            g_data, momentum, sign_momentum, exp_avg_sq,
            beta1, beta2, beta3, step_t, eps,
        )
        poly_beta3 = _poly_beta(beta3, step_t)

        # 6. Warp the crafted update (if any). No orthogonalization for 1D.
        if spectral_log_left is not None:
            update_2d = update.reshape(-1, 1)
            warped = _spectral_apply(update_2d, spectral_log_left, None, spectral_log_bound)
            O = torch.where(step_t <= 1.0, update_2d, warped).reshape(update.shape)
        elif warp is not None:
            update_2d = update.reshape(-1, 1)
            O = (update_2d + warp.float() @ update_2d).reshape(update.shape)
        else:
            O = update

        # 7-8. Variance-normalize + rescale back to the warped update's RMS.
        update_final, target_rms = _normalize_and_rescale(O, row_var, poly_beta3, eps, O)

        # 9. Cautious + decoupled weight decay.
        update_final = _cautious(update_final, g_data, cautious_update)
        update_final = apply_weight_decay(
            p_data, update_final, g_data, weight_decay, eps,
            cautious_update, cautious_wd, relative_wd, target_rms,
        )

        return update_final, update

    # -----------------------------------------------------------------------
    # State init + step dispatch (shared helpers)
    # -----------------------------------------------------------------------

    def _ensure_state(self, p, group) -> None:
        """Lazily create all optimizer state for ``p`` (idempotent)."""
        state = self.state[p]
        if len(state) != 0:
            return

        state["step"] = 0
        state["step_t"] = torch.tensor(0.0, dtype=torch.float32, device=p.device)
        if p.ndim >= 2:
            w_2d = _reshape_to_2d(p.data)
            m, n = w_2d.shape
            state["momentum"] = torch.zeros_like(w_2d, dtype=torch.float32)
            state["sign_momentum"] = torch.zeros_like(w_2d, dtype=torch.float32)
            state["exp_avg_sq_row"] = torch.zeros((m, 1), device=p.device, dtype=torch.float32)
            state["exp_avg_sq_col"] = torch.zeros((1, n), device=p.device, dtype=torch.float32)
            state["row_var"] = torch.zeros(m, device=p.device, dtype=torch.float32)
            warp_m = m
        else:
            state["momentum"] = torch.zeros_like(p.data, dtype=torch.float32)
            state["sign_momentum"] = torch.zeros_like(p.data, dtype=torch.float32)
            state["exp_avg_sq"] = torch.zeros_like(p.data, dtype=torch.float32)
            state["row_var"] = torch.zeros((), device=p.device, dtype=torch.float32)
            warp_m = p.numel()

        if group["meta_lr"] > 0:
            if group["warp_mode"] == "dense":
                state["warp"] = torch.zeros(
                    warp_m, warp_m, dtype=group["warp_dtype"], device=p.device,
                )
            else:
                state["spectral_log_left"] = torch.zeros(
                    warp_m, dtype=torch.float32, device=p.device,
                )
                if p.ndim >= 2 and group["spectral_bilateral"]:
                    state["spectral_log_right"] = torch.zeros(
                        w_2d.shape[1], dtype=torch.float32, device=p.device,
                    )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError(
                        "WarpAINO does not support sparse gradients"
                    )
                self._ensure_state(p, group)

            if self._foreach:
                self._step_foreach(group)
            else:
                self._step_native(group)

        return loss

    @staticmethod
    def _copy_back(p, p_fp32, use_view, stochastic_fp):
        """Copy the updated fp32 buffer back into ``p`` (skipped for views)."""
        if use_view:
            return
        p_reshaped = p_fp32.view_as(p.data)
        if p.dtype is torch.bfloat16 and stochastic_fp:
            copy_stochastic_(p.data, p_reshaped)
        else:
            p.data.copy_(p_reshaped)

    def _run_core(self, p, group, state):
        """Run the unified core for one parameter.

        Performs step bookkeeping, dtype promotion (``use_view``), and the core
        call, but does **not** apply ``-lr`` or copy back -- the caller does
        that (native per-param, foreach batched). Returns
        ``(p_fp32, use_view, update_final, meta)`` where ``meta`` is either
        ``None`` or ``("spectral"|"warp", state, crafted_update)``.
        """
        lr = group["lr"]
        beta1, beta2, beta3 = group["betas"]
        weight_decay = group["weight_decay"]
        cautious_update = group["cautious_update"]
        cautious_wd = group["cautious_wd"]
        stochastic_fp = group["stochastic_fp"]
        sinkhorn_steps = group["sinkhorn_steps"]
        ortho_dtype = group["ortho_dtype"]
        eps = group["eps"]
        meta_lr = group["meta_lr"]
        meta_wd = group["meta_wd"]
        spectral_bilateral = group["spectral_bilateral"]
        spectral_log_bound = group["spectral_log_bound"]
        relative_wd = group.get("relative_wd", False)

        # Shared bookkeeping.
        state["step"] += 1
        if "step_t" not in state:
            state["step_t"] = torch.tensor(float(state["step"]), dtype=torch.float32, device=p.device)
        else:
            state["step_t"].fill_(float(state["step"]))
        step_t = state["step_t"]

        grad = p.grad.data
        if grad.dtype in (torch.bfloat16, torch.float16):
            grad = grad.float()

        warp = state.get("warp")
        spectral_log_left = state.get("spectral_log_left")
        spectral_log_right = state.get("spectral_log_right")
        if spectral_log_right is None and spectral_log_left is not None:
            spectral_log_right = torch.empty(0, dtype=torch.float32, device=p.device)

        if p.ndim >= 2:
            p_2d = _reshape_to_2d(p.data)
            g_2d = _reshape_to_2d(grad)
            use_view = (p.dtype is torch.float32) and p.data.is_contiguous()
            p_2d_fp32 = p_2d if use_view else (p_2d.float() if p.dtype is torch.bfloat16 else p_2d.clone())

            update_final, crafted_update = self._compiled_2d(
                p_2d_fp32, g_2d, state["momentum"], state["sign_momentum"],
                state["exp_avg_sq_row"], state["exp_avg_sq_col"], state["row_var"],
                beta1, beta2, beta3, weight_decay, eps, step_t, sinkhorn_steps,
                cautious_update, cautious_wd, ortho_dtype, relative_wd,
                warp, spectral_log_left, spectral_log_right, spectral_bilateral, spectral_log_bound,
            )

            meta = None
            if spectral_log_left is not None:
                meta = ("spectral", state, crafted_update)
            elif warp is not None:
                meta = ("warp", state, crafted_update)
            return p_2d_fp32, use_view, update_final, meta
        else:
            use_view = (p.dtype is torch.float32) and p.data.is_contiguous()
            p_fp32 = p.data if use_view else (p.data.float() if p.dtype is torch.bfloat16 else p.data.clone())

            update_final, crafted_update = self._compiled_1d(
                p_fp32, grad, state["momentum"], state["sign_momentum"],
                state["exp_avg_sq"], state["row_var"],
                beta1, beta2, beta3, weight_decay, eps, step_t,
                cautious_update, cautious_wd, relative_wd,
                warp, spectral_log_left, spectral_log_bound,
            )

            meta = None
            if spectral_log_left is not None:
                meta = ("spectral", state, crafted_update.reshape(-1, 1))
            elif warp is not None:
                meta = ("warp", state, crafted_update.reshape(-1, 1))
            return p_fp32, use_view, update_final, meta

    @staticmethod
    def _run_meta(meta, meta_lr, meta_wd, spectral_log_bound):
        """Apply the per-parameter meta-update recorded by ``_run_core``."""
        if meta is None:
            return
        kind, state, crafted_update = meta
        if kind == "spectral":
            _spectral_meta_update(state, crafted_update, meta_lr, meta_wd, spectral_log_bound)
        else:
            _warp_meta_update(state, crafted_update, meta_lr, meta_wd)

    def _step_native(self, group):
        meta_lr = group["meta_lr"]
        meta_wd = group["meta_wd"]
        spectral_log_bound = group["spectral_log_bound"]
        stochastic_fp = group["stochastic_fp"]

        for p in group["params"]:
            if p.grad is None:
                continue
            p_fp32, use_view, update_final, meta = self._run_core(p, group, self.state[p])
            p_fp32.add_(update_final, alpha=-group["lr"])
            self._copy_back(p, p_fp32, use_view, stochastic_fp)
            self._run_meta(meta, meta_lr, meta_wd, spectral_log_bound)

    def _step_foreach(self, group):
        meta_lr = group["meta_lr"]
        meta_wd = group["meta_wd"]
        spectral_log_bound = group["spectral_log_bound"]
        stochastic_fp = group["stochastic_fp"]

        params_1d = [p for p in group["params"] if p.grad is not None and p.ndim < 2]
        params_2d = [p for p in group["params"] if p.grad is not None and p.ndim >= 2]

        # 1. 1D/0D parameters: processed individually (no batching benefit).
        for p in params_1d:
            p_fp32, use_view, update_final, meta = self._run_core(p, group, self.state[p])
            p_fp32.add_(update_final, alpha=-group["lr"])
            self._copy_back(p, p_fp32, use_view, stochastic_fp)
            self._run_meta(meta, meta_lr, meta_wd, spectral_log_bound)

        # 2. 2D+ parameters: batch the final ``-lr`` add for throughput.
        if not params_2d:
            return

        updates_final_list = []
        p_fp32_list = []
        p_original_list = []
        use_view_list = []
        meta_list = []

        for p in params_2d:
            p_fp32, use_view, update_final, meta = self._run_core(p, group, self.state[p])
            updates_final_list.append(update_final.view_as(p_fp32))
            p_fp32_list.append(p_fp32)
            p_original_list.append(p)
            use_view_list.append(use_view)
            meta_list.append(meta)

        torch._foreach_add_(p_fp32_list, updates_final_list, alpha=-group["lr"])

        for p, p_fp32, use_view in zip(p_original_list, p_fp32_list, use_view_list):
            self._copy_back(p, p_fp32, use_view, stochastic_fp)

        for meta in meta_list:
            self._run_meta(meta, meta_lr, meta_wd, spectral_log_bound)
