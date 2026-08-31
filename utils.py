#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import os

import torch

from c_functions import PI, icdf_gauss, _c_div, fft_idx2digfreq, fft_idx2digfreq_t, digfreq2fft_idx
from input_output import write_mrc

MAX_RESOLUTION_CANDIDATES = 100000    # safety cap


# ---------------------------------------------------------------------------
# Geometry / frequency grids
# ---------------------------------------------------------------------------

def xmipp_centered_radius2(shape, device) -> torch.Tensor:
    """``k*k + i*i + j*j`` on the Xmipp centred logical grid.

    Mirrors ``FOR_ALL_ELEMENTS_IN_ARRAY3D`` after ``setXmippOrigin()``: along an
    axis of size ``N`` the logical index of array position ``n`` is
    ``n - (N // 2)`` (``FIRST_XMIPP_INDEX``).
    """
    nz, ny, nx = shape
    k = torch.arange(nz, device=device, dtype=torch.float64) - (nz // 2)
    i = torch.arange(ny, device=device, dtype=torch.float64) - (ny // 2)
    j = torch.arange(nx, device=device, dtype=torch.float64) - (nx // 2)
    return (k * k).view(-1, 1, 1) + (i * i).view(1, -1, 1) + (j * j).view(1, 1, -1)


def fourier_freqs_3d(real_shape, device):
    """Port of ``Monogenic::fourierFreqs_3D`` (+ ``fourierFreqVector``).

    Returns ``(iu, freq_fourier_x, freq_fourier_y, freq_fourier_z)`` where ``iu``
    is the ``(nz, ny, nx//2+1)`` map of ``1/|w|`` and the three vectors hold the
    per-axis digital frequencies.  Element 0 of every vector is the ``1e-38``
    sentinel and ``iu[0,0,0]`` is ``1e38``, exactly as in the C++.
    """
    nz, ny, nx = real_shape
    fz_n, fy_n, fx_n = nz, ny, nx // 2 + 1     # half-spectrum dimensions

    def _freq_vector(n_fourier: int, n_real: int) -> torch.Tensor:
        idx = torch.arange(n_fourier, device=device)
        vec = fft_idx2digfreq_t(idx, n_real)
        vec[0] = 1e-38                          # "a really low value to represent 0"
        return vec

    freq_fourier_z = _freq_vector(fz_n, nz)
    freq_fourier_y = _freq_vector(fy_n, ny)
    freq_fourier_x = _freq_vector(fx_n, nx)

    # iu is built from the macro directly (NOT from the vectors above), so no
    # 1e-38 sentinel leaks into it.
    uz = fft_idx2digfreq_t(torch.arange(fz_n, device=device), nz).view(-1, 1, 1)
    uy = fft_idx2digfreq_t(torch.arange(fy_n, device=device), ny).view(1, -1, 1)
    ux = fft_idx2digfreq_t(torch.arange(fx_n, device=device), nx).view(1, 1, -1)
    u2 = (uz * uz + uy * uy + ux * ux).expand(fz_n, fy_n, fx_n).contiguous()
    # u2 == 0 only at DC, which is overwritten right below; the substitution
    # avoids a spurious division warning there.
    iu = 1.0 / torch.sqrt(torch.where(u2 > 0, u2, torch.ones_like(u2)))
    iu[0, 0, 0] = 1e38                          # "infinite" inverse frequency at DC

    return iu, freq_fourier_x, freq_fourier_y, freq_fourier_z


def real_gaussian_filter(volume: torch.Tensor, sigma: float) -> torch.Tensor:
    """Port of ``realGaussianFilter`` (``libraries/data/fourier_filter.cpp``).

    Xmipp implements it as a *Fourier-domain* multiplication by
    ``exp(-PI*PI*|w|^2*sigma^2)`` (``FourierFilter::maskValue``, case
    ``REALGAUSSIAN``), where ``|w|`` is the digital frequency modulus built from
    ``FFT_IDX2DIGFREQ``.  Reproduced literally here.
    """
    nz, ny, nx = volume.shape
    device = volume.device
    spec = torch.fft.rfftn(volume, dim=(-3, -2, -1))
    uz = fft_idx2digfreq_t(torch.arange(nz, device=device), nz).view(-1, 1, 1)
    uy = fft_idx2digfreq_t(torch.arange(ny, device=device), ny).view(1, -1, 1)
    ux = fft_idx2digfreq_t(torch.arange(nx // 2 + 1, device=device), nx).view(1, 1, -1)
    w2 = uz * uz + uy * uy + ux * ux
    spec = spec * torch.exp(-(PI * PI) * w2 * (sigma * sigma))
    return torch.fft.irfftn(spec, s=(nz, ny, nx), dim=(-3, -2, -1))


# ---------------------------------------------------------------------------
# Mask preparation
# ---------------------------------------------------------------------------

def protein_radius_and_volume(mask: torch.Tensor):
    """Port of ``Monogenic::proteinRadiusVolumeAndShellStatistics``.

    Returns ``(radius, n_voxels)``: the truncated integer radius of the farthest
    ``mask == 1`` voxel from the centre, and the number of ``mask == 1`` voxels.
    """
    r2 = xmipp_centered_radius2(mask.shape, mask.device)
    sel = mask == 1
    n_voxels = int(sel.sum().item())
    max_r2 = float(r2[sel].max().item()) if n_voxels > 0 else 0.0
    radius = int(math.sqrt(max_r2))            # C++: int radius = sqrt(int R2max)
    print("                                     ")
    print("The protein has a radius of %d px " % radius)
    return radius, n_voxels


def find_cliff_value(input_vol: torch.Tensor, radius: int, mask: torch.Tensor,
                     rsmooth: float = 0.0) -> int:
    """Port of ``Monogenic::findCliffValue``.  Mutates ``mask`` in place.

    Grows unit-thickness spherical shells outwards from ``radius`` and stops as
    soon as the shell variance collapses (< 1% of the previous shell), which is
    where a spherical mask was applied to the map and beyond which there is no
    noise.  Everything with ``r^2 >= (radiuslimit - rsmooth)^2`` is then flagged
    with ``-1`` in the mask.
    """
    _criticalZ = icdf_gauss(0.95)              # computed by the C++ but unused there
    nx = input_vol.shape[2]
    radiuslimit = nx // 2

    r2 = xmipp_centered_radius2(input_vol.shape, input_vol.device)

    last_std2 = 1e-38                          # C++ seed, avoids 0/0 on rad == radius
    for rad in range(radius, radiuslimit):
        inf = float(rad * rad)
        sup = float((rad + 1) * (rad + 1))
        shell = (r2 < sup) & (r2 >= inf)
        n = float(shell.sum().item())
        if n == 0.0:
            # C++ divides by N == 0 and carries NaN forward from here on; the
            # `< 0.01` test is then always false. Reproduced (cannot happen for
            # rad < XSIZE/2 on a centred grid, but kept for exactness).
            mean = float("nan")
            std2 = float("nan")
        else:
            values = input_vol[shell]
            s = float(values.sum().item())
            s2 = float((values * values).sum().item())
            mean = s / n
            std2 = s2 / n - mean * mean
        if _c_div(std2, last_std2) < 0.01:
            radiuslimit = rad - 1
            break
        last_std2 = std2

    print("There is no noise beyond a radius of %d px " % radiuslimit)
    print("Regions with a radius greater than %d px will not be considered" % radiuslimit)

    raux = float(radiuslimit) - float(rsmooth)
    if raux <= radius:
        print("Warning: the boxsize is very close to the protein size "
              "please provide a greater box")
    raux *= raux

    mask[r2 >= raux] = -1
    return radiuslimit


def exclude_area(mask: torch.Tensor, mask_excl: torch.Tensor,
                 half_maps_given: bool, noise_only_in_halves: bool) -> None:
    """Port of ``ProgMonogenicSignalRes::excludeArea``.  Mutates ``mask``.

    Careful: in half-map mode *without* ``--noiseonlyinhalves`` the C++ does
    nothing at all (the ``if (noiseOnlyInHalves)`` has no ``else``), i.e.
    ``--maskExcl`` is silently ignored.  Replicated.
    """
    if half_maps_given:
        if noise_only_in_halves:
            inside = mask == 1
            mask[inside & (mask_excl == 1)] = -1
            mask[~inside] = -1
        # else: nothing (faithful to the C++)
    else:
        mask[(mask == 1) & (mask_excl == 1)] = -1


# ---------------------------------------------------------------------------
# Monogenic amplitude
# ---------------------------------------------------------------------------

def monogenic_amplitude_3d_fourier(myfftV: torch.Tensor, iu: torch.Tensor,
                                   fx: torch.Tensor, fy: torch.Tensor,
                                   fz: torch.Tensor, real_shape) -> torch.Tensor:
    """Port of ``Monogenic::monogenicAmplitude_3D_Fourier``.

    Full-band monogenic amplitude: same Riesz maths as
    :func:`amplitude_mono_sig_3d_lpf` but with *no* band-pass filtering and no
    final low-pass smoothing.  Used by :func:`refine_mask`.

    The DC bin is where the two Xmipp sentinels meet: ``iu[0,0,0] == 1e38``
    blows the DC coefficient up and ``freq_fourier_*[0] == 1e-38`` brings it
    back down, so the product is ~1x the DC coefficient.  Do not "simplify".
    """
    nz, ny, nx = real_shape
    F = myfftV
    F_aux = (-1j) * F * iu

    amplitude = torch.fft.irfftn(F, s=(nz, ny, nx), dim=(-3, -2, -1))
    amplitude = amplitude * amplitude

    for u in (fx.view(1, 1, -1), fy.view(1, -1, 1), fz.view(-1, 1, 1)):
        comp = torch.fft.irfftn(u * F_aux, s=(nz, ny, nx), dim=(-3, -2, -1))
        amplitude = amplitude + comp * comp

    return torch.sqrt(amplitude)


def amplitude_mono_sig_3d_lpf(myfftV: torch.Tensor, freq: float, freqH: float,
                              freqL: float, iu: torch.Tensor, fx: torch.Tensor,
                              fy: torch.Tensor, fz: torch.Tensor,
                              real_shape) -> torch.Tensor:
    """Port of ``Monogenic::amplitudeMonoSig3D_LPF``.

    (a) raised-cosine high-pass of ``myfftV`` over ``[freqH, freq]`` (pass-through
        above ``freq``, zero below ``freqH``) -> ``F``, and ``F_aux = -i * F * iu``
    (b) ``amplitude = |IFFT(F)|^2``                          (the I0 component)
    (c,d) add the squares of the three Riesz components ``IFFT(u_a * F_aux)``,
        all built from the *same* ``F_aux``
    (e) ``amplitude = sqrt(...)``
    (f) raised-cosine low-pass of the amplitude itself over ``[freq, freqL]``

    Note on step (f): when ``freq`` snaps exactly onto Nyquist, ``freqL`` is
    clamped to 0.5 so the ramp width is 0 and ``PI/0 == inf``; voxels with
    ``|w| == 0.5`` then get ``cos(inf*0) == NaN``.  The C++ does exactly the same
    (see monores.py deviation #7) and we do not special-case it.
    """
    nz, ny, nx = real_shape
    un = 1.0 / iu                       # C++: double un = 1.0/iun;

    # ---- (a) high-pass raised cosine ------------------------------------
    ideltal = _c_div(PI, freq - freqH)
    ramp = 0.5 * (1.0 + torch.cos((un - freq) * ideltal))
    in_ramp = (un >= freqH) & (un <= freq)
    above = un > freq
    gain = torch.where(in_ramp, ramp, torch.zeros_like(ramp))
    gain = torch.where(above, torch.ones_like(gain), gain)
    F = myfftV * gain
    F_aux = (-1j) * F * iu

    # ---- (b) I0 component ------------------------------------------------
    amplitude = torch.fft.irfftn(F, s=(nz, ny, nx), dim=(-3, -2, -1))
    amplitude = amplitude * amplitude

    # ---- (c)+(d) the three Riesz components, all from the same F_aux -----
    for u in (fx.view(1, 1, -1), fz.view(-1, 1, 1), fy.view(1, -1, 1)):
        comp = torch.fft.irfftn(u * F_aux, s=(nz, ny, nx), dim=(-3, -2, -1))
        amplitude = amplitude + comp * comp

    # ---- (e) --------------------------------------------------------------
    amplitude = torch.sqrt(amplitude)

    # ---- (f) low-pass smoothing of the amplitude -------------------------
    spec = torch.fft.rfftn(amplitude, dim=(-3, -2, -1))
    raised_w = _c_div(PI, freqL - freq)
    lp_ramp = 0.5 * (1.0 + torch.cos(raised_w * (un - freq)))
    in_lp = (un <= freqL) & (un >= freq)
    over = un > freqL
    lp_gain = torch.where(in_lp, lp_ramp, torch.ones_like(lp_ramp))
    lp_gain = torch.where(over, torch.zeros_like(lp_gain), lp_gain)
    spec = spec * lp_gain
    return torch.fft.irfftn(spec, s=(nz, ny, nx), dim=(-3, -2, -1))


def refine_mask(myfftV: torch.Tensor, iu: torch.Tensor, fx, fy, fz,
                mask: torch.Tensor, real_shape):
    """Port of ``ProgMonogenicSignalRes::refiningMask``.  Mutates ``mask``.

    Computes the full-band monogenic amplitude, smooths it with
    ``realGaussianFilter(sigma=4)``, takes the 95th percentile (nearest rank) of
    the amplitude over the noise region (``mask == 0``) and drops every
    ``mask >= 1`` voxel whose smoothed amplitude falls below it.

    Returns the refreshed ``NVoxelsOriginalMask``.
    """
    amplitude = monogenic_amplitude_3d_fourier(myfftV, iu, fx, fy, fz, real_shape)
    amplitude = real_gaussian_filter(amplitude, 4.0)   # "std = 4"

    signal_sel = mask >= 1
    noise_sel = mask == 0

    n_s = int(signal_sel.sum().item())
    n_n = int(noise_sel.sum().item())
    if n_n == 0:
        raise RuntimeError("refiningMask: the noise region (mask == 0) is empty; "
                           "the provided mask leaves no background voxels.")
    if n_s == 0:
        raise RuntimeError("refiningMask: the signal region (mask >= 1) is empty.")

    _mean_signal = float(amplitude[signal_sel].sum().item()) / n_s
    noise_values = amplitude[noise_sel]
    _mean_noise = float(noise_values.sum().item()) / n_n

    noise_sorted, _ = torch.sort(noise_values)
    thr_idx = int(n_n * 0.95)                    # size_t(size*0.95), truncating
    thr_idx = min(thr_idx, n_n - 1)
    threshold_first_estimation = float(noise_sorted[thr_idx].item())

    drop = signal_sel & (amplitude < threshold_first_estimation)
    mask[drop] = 0
    n_voxels_original_mask = int((mask >= 1).sum().item())
    return n_voxels_original_mask


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _mean_var(values: torch.Tensor):
    """``mean``, ``sum(x^2)/N - mean^2`` exactly as the C++ accumulates them."""
    n = float(values.numel())
    s = float(values.sum().item())
    s2 = float((values * values).sum().item())
    mean = s / n
    return mean, s2 / n - mean * mean, n


def _nearest_rank(values: torch.Tensor, significance: float) -> float:
    """``sorted[size_t(N*significance)]`` -- nearest rank, not interpolated."""
    n = values.numel()
    srt, _ = torch.sort(values)
    idx = int(n * significance)
    if idx >= n:
        idx = n - 1
    return float(srt[idx].item())


def statistics_in_binary_mask(volS: torch.Tensor, volN: torch.Tensor,
                              mask: torch.Tensor, significance: float):
    """Port of ``Monogenic::statisticsInBinaryMask2`` (half-map mode).

    Signal statistics over ``mask > 0`` of ``volS``; noise statistics and the
    ``thr95`` percentile over ``mask >= 0`` of ``volN``.  (Yes, ``>= 0`` -- the
    C++ even has a "BE CAREFULL WITH THE =" comment.)
    """
    sig = volS[mask > 0]
    noi = volN[mask >= 0]
    if sig.numel() == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    if noi.numel() == 0:
        raise RuntimeError("statisticsInBinaryMask2: no noise voxels (mask >= 0).")
    meanS, sdS2, NS = _mean_var(sig)
    meanN, sdN2, NN = _mean_var(noi)
    thr95 = _nearest_rank(noi, significance)
    return meanS, sdS2, meanN, sdN2, thr95, NS, NN


def statistics_in_out_binary_mask(volS: torch.Tensor, mask: torch.Tensor,
                                  significance: float):
    """Port of ``Monogenic::statisticsInOutBinaryMask2`` (single-map mode).

    Signal statistics over ``mask >= 1`` and noise statistics over ``mask == 0``,
    both taken from the *same* amplitude volume ``volS``.
    """
    sig = volS[mask >= 1]
    noi = volS[mask == 0]
    if sig.numel() == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    if noi.numel() == 0:
        raise RuntimeError("statisticsInOutBinaryMask2: no noise voxels (mask == 0).")
    meanS, sdS2, NS = _mean_var(sig)
    meanN, sdN2, NN = _mean_var(noi)
    thr95 = _nearest_rank(noi, significance)
    return meanS, sdS2, meanN, sdN2, thr95, NS, NN


# ---------------------------------------------------------------------------
# Local-resolution assignment
# ---------------------------------------------------------------------------

def set_local_resolution(amplitudeMS: torch.Tensor, mask: torch.Tensor,
                         resolution_map: torch.Tensor, threshold_noise: float,
                         resolution: float, resolution_2: float,
                         half_maps_given: bool) -> None:
    """Port of ``Monogenic::setLocalResolutionHalfMaps`` /
    ``Monogenic::setLocalResolutionMap`` (identical except the "give up" mask
    value: 0 for half maps, -1 for a single map).  Mutates ``mask`` and
    ``resolution_map``.

    The mask doubles as a per-voxel failure counter: a voxel whose amplitude
    dips below the noise threshold gets ``mask += 1`` and is only finalised (at
    ``resolution_2``, i.e. two frequency steps back) once it has missed three
    times in a row (``mask > 2``).  This hysteresis is part of the published
    algorithm -- do not simplify it away.
    """
    sel = mask >= 1
    passed = sel & (amplitudeMS > threshold_noise)       # NaN -> False, as in C++
    failed = sel & ~(amplitudeMS > threshold_noise)

    mask[passed] = 1
    resolution_map[passed] = resolution

    mask[failed] += 1
    give_up = failed & (mask > 2)
    mask[give_up] = 0 if half_maps_given else -1
    resolution_map[give_up] = resolution_2


# ---------------------------------------------------------------------------
# The frequency sweep
# ---------------------------------------------------------------------------

def resolution2eval(freq_step: float, sampling: float, max_res: float,
                    volsize: int):
    """Generator port of ``Monogenic::resolution2eval`` plus the ``do { ... }
    while (doNextIteration)`` ``continue``/``break`` plumbing in
    ``ProgMonogenicSignalRes::run()``.

    Yields ``(resolution, freq, freqL)`` for every *accepted* candidate.
    Candidates that snap onto the previous Fourier index are skipped silently
    (the C++ ``continueIter`` path, which does not advance ``iter`` nor append to
    ``list``); the generator ends when the candidate resolution falls below
    Nyquist (the C++ ``breakIter`` path).

    ``volsize`` is ``XSIZE(pMask)`` -- the C++ carries a ``//TODO: take minimum
    size`` here and really does use the X size for all three axes, i.e. it
    assumes a cubic box.  Replicated as-is, deliberately not "fixed".
    """
    count_res = 0
    last_fourier_idx = -1
    nyquist = 2.0 * sampling
    guard = 0

    while True:
        guard += 1
        if guard > MAX_RESOLUTION_CANDIDATES:
            print("Warning: resolution2eval exceeded %d candidate frequencies; "
                  "stopping the sweep (see monores.py deviation #5)." % MAX_RESOLUTION_CANDIDATES)
            return

        resolution = max_res - count_res * freq_step
        freq = _c_div(sampling, resolution)
        count_res += 1

        try:
            fourier_idx = digfreq2fft_idx(freq, volsize)
        except ValueError:
            # freq is inf/NaN (resolution hit exactly 0): the C++ would feed
            # garbage to round(); nothing sensible can follow, so stop.
            return

        aux_frequency = fft_idx2digfreq(fourier_idx, volsize)
        freq = aux_frequency

        if fourier_idx == last_fourier_idx:
            continue                                  # C++ continueIter

        last_fourier_idx = fourier_idx
        resolution = _c_div(sampling, aux_frequency)

        # (dead code omitted: `if (count_res == 0) last_resolution = resolution;`
        #  -- count_res was just incremented, so it can never be 0 here.)

        if resolution < nyquist:
            return                                    # C++ breakIter

        freqL = _c_div(sampling, resolution + freq_step)
        try:
            fourier_idx_2 = digfreq2fft_idx(freqL, volsize)
        except ValueError:
            fourier_idx_2 = None
        if fourier_idx_2 == fourier_idx:
            if fourier_idx > 0:
                freqL = fft_idx2digfreq(fourier_idx - 1, volsize)
            else:
                freqL = _c_div(sampling, resolution + freq_step)

        # NOTE: run() passes this freqL into its own `freqH` variable and
        # overwrites it two lines later, so it never reaches
        # amplitudeMonoSig3D_LPF. Yielded for traceability only.
        yield resolution, freq, freqL


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def post_process_local_resolutions(filtered_map: torch.Tensor,
                                   resolution_vol: torch.Tensor,
                                   list_res, mask: torch.Tensor,
                                   sampling: float, out_dir: str) -> None:
    """Port of ``ProgMonogenicSignalRes::postProcessingLocalResolutions``.

    Unresolved voxels (final resolution better than the last swept resolution,
    which in practice means the untouched zeros) are filled with the median of
    the *worse-than-last* population and masked out; the map is then smoothed
    with ``realGaussianFilter(sigma=3)`` and the smoothed value is rolled back to
    the raw estimate wherever smoothing made it worse or pushed it below
    Nyquist.  Writes ``refinedMask.mrc``, ``monoresResolutionChimera.mrc`` and
    ``monoresResolutionMap.mrc``.
    """
    nyquist = 2.0 * sampling

    last_res = list_res[-1] - 0.001            # 0.001 is the C++ tolerance
    n_count = int((filtered_map >= last_res).sum().item())
    values = filtered_map[filtered_map > last_res]
    if n_count == 0 or values.numel() == 0:
        raise RuntimeError("postProcessingLocalResolutions: no voxel reached the "
                           "last analysed resolution (%.4f A); nothing to write."
                           % list_res[-1])
    # The C++ allocates N entries counted with '>=' but fills them with '>'.
    # With discrete resolution values the two counts coincide; if they ever did
    # not, the C++ would sort uninitialised (calloc'ed => 0) tail entries.
    if values.numel() < n_count:
        pad = torch.zeros(n_count - values.numel(), dtype=values.dtype,
                          device=values.device)
        values = torch.cat([values, pad])
    values, _ = torch.sort(values)
    filling_value = float(values[int(0.5 * n_count)].item())   # median

    last_res = list_res[-1]
    bad = filtered_map < last_res
    filtered_map[bad] = filling_value
    mask[bad] = 0
    mask[~bad] = 1

    write_mrc(os.path.join(out_dir, "refinedMask.mrc"), mask.to(torch.float64), sampling)

    filtered_map = real_gaussian_filter(filtered_map, 3.0)

    # Smoothing may only improve (lower) the resolution value, and never dive
    # below Nyquist; otherwise fall back to the raw per-voxel estimate.
    inside = mask > 0
    revert = inside & ((filtered_map > resolution_vol) | (filtered_map < nyquist))
    filtered_map = torch.where(revert, resolution_vol, filtered_map)

    write_mrc(os.path.join(out_dir, "monoresResolutionChimera.mrc"),
              filtered_map, sampling)

    filtered_map = torch.where(mask == 0, torch.zeros_like(filtered_map), filtered_map)
    write_mrc(os.path.join(out_dir, "monoresResolutionMap.mrc"), filtered_map, sampling)
