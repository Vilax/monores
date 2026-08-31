#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MonoRes local resolution estimation
===================================================================

MonoRes estimates a local resolution map of a cryo-EM reconstruction (or of a
pair of half maps) by comparing, band by band, the monogenic amplitude of the
signal against the monogenic amplitude of the noise through a one-sided
hypothesis test.

Reference:

    J.L. Vilas et al., "MonoRes: Automatic and Accurate Estimation of Local
    Resolution for Electron Microscopy Maps", Structure, 26, 337-344 (2018).

Authors:
Jose Luis Vilas (jlvilas@cnb.csic.es
Carlos Oscar S. Sorzano (coss@cnb.csic.es)

Outputs:

* ``meanMap.mrc``                 (half-map mode only) 0.5*(V1+V2)
* ``refinedMask.mrc``             binary mask of voxels with a valid resolution
* ``monoresResolutionChimera.mrc`` smoothed resolution map, full box
* ``monoresResolutionMap.mrc``    smoothed resolution map, masked
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

from c_functions import DBL_MIN, icdf_gauss, _c_div, _c_sqrt
from input_output import read_mrc, write_mrc
from utils import (
    fourier_freqs_3d,
    protein_radius_and_volume,
    find_cliff_value,
    exclude_area,
    refine_mask,
    amplitude_mono_sig_3d_lpf,
    statistics_in_binary_mask,
    statistics_in_out_binary_mask,
    set_local_resolution,
    resolution2eval,
    post_process_local_resolutions,
)


def run(args) -> None:
    device = torch.device(args.device)
    dt = torch.float64

    os.makedirs(args.output, exist_ok=True)

    print("Starting...")

    # ---------------- produceSideInfo -----------------------------------
    if args.vol and args.vol2:
        v1 = torch.as_tensor(read_mrc(args.vol), dtype=dt, device=device)
        v2 = torch.as_tensor(read_mrc(args.vol2), dtype=dt, device=device)
        if v1.shape != v2.shape:
            raise RuntimeError("--vol and --vol2 have different dimensions")
        input_vol = 0.5 * (v1 + v2)
        write_mrc(os.path.join(args.output, "meanMap.mrc"), input_vol,
                  args.sampling_rate)
        noise_vol = (v1 - v2) / 2.0
        fftN = torch.fft.rfftn(noise_vol, dim=(-3, -2, -1))
        half_maps_given = True
        del v1, v2, noise_vol
    else:
        input_vol = torch.as_tensor(read_mrc(args.vol), dtype=dt, device=device)
        half_maps_given = False
        fftN = None           

    real_shape = tuple(input_vol.shape)
    nz, ny, nx = real_shape


    if not args.mask:
        print("Error: a mask ought to be provided")
        sys.exit(0)
    mask_np = read_mrc(args.mask)
    if mask_np.shape != real_shape:
        raise RuntimeError("--mask dimensions %s do not match the volume %s"
                           % (mask_np.shape, real_shape))
    mask = torch.as_tensor(np.trunc(mask_np).astype(np.int64), dtype=torch.int64,
                           device=device)

    smoothparam = 0.0
    radius, n_voxels_original_mask = protein_radius_and_volume(mask)
    find_cliff_value(input_vol, radius, mask, smoothparam)

    if args.maskExcl:
        excl_np = read_mrc(args.maskExcl)
        if excl_np.shape != real_shape:
            raise RuntimeError("--maskExcl dimensions do not match the volume")
        mask_excl = torch.as_tensor(np.trunc(excl_np).astype(np.int64),
                                    dtype=torch.int64, device=device)
        exclude_area(mask, mask_excl, half_maps_given, args.noiseonlyinhalves)
        del mask_excl
    else:
        if half_maps_given and args.noiseonlyinhalves:
            mask[mask < 1] = -1

    fftV = torch.fft.rfftn(input_vol, dim=(-3, -2, -1))

    # Frequency volume
    iu, fx, fy, fz = fourier_freqs_3d(real_shape, device)

    freq_step = args.step
    if freq_step < 0.25:
        freq_step = 0.25

    del input_vol

    # ---------------- run ------------------------------------------------
    critical_z = icdf_gauss(args.significance)
    max_meanS = -DBL_MIN
    cut_value = 0.025                     # cut_value represents a percentile 2.5
    sampling = args.sampling_rate
    min_res = 2.0 * sampling              # the CLI --minRes is overwritten here
    max_res = args.maxRes

    volsize = nx

    # The sweep starts at --maxRes and decreases towards Nyquist, 
    # so --maxRes must be the LOW resolution end.  With the stock 
    # default (--maxRes 1) the first candidate frequency aliases 
    # onto the DC bin and the C++ obtains resolution == inf,
    # which then poisons the output map.  Warn loudly instead of silently
    # producing a NaN map.
    if max_res <= min_res:
        print("Warning: --maxRes (%g A) is not above Nyquist (%g A). The sweep "
              "runs from --maxRes DOWNWARDS to Nyquist, so --maxRes must be the "
              "low-resolution start of the range (e.g. 15-30 A). Results will "
              "be meaningless." % (max_res, min_res))

    if not args.noiseonlyinhalves:
        n_voxels_original_mask = refine_mask(fftV, iu, fx, fy, fz, mask, real_shape)

    if n_voxels_original_mask == 0:
        raise RuntimeError("The mask contains no protein voxels after refinement.")

    resolution_map = torch.zeros(real_shape, dtype=dt, device=device)
    list_res = []
    iteration = 0
    do_next_iteration = True
    lefttrimming = False
    n_voxels = 0

    print("Analyzing frequencies")

    for resolution, freq, _freqL_from_resolution2eval in resolution2eval(
            freq_step, sampling, max_res, volsize):

        print("resolution = %s" % resolution)

        list_res.append(resolution)
        resolution_2 = list_res[0] if iteration < 2 else list_res[iteration - 2]

        # 0.02 is the tail of the raise cosine in digital units.  NOTE the
        # variable-name swap versus resolution2eval: these are the values that
        # actually reach amplitudeMonoSig3D_LPF.
        freqL = freq + 0.02
        freqH = freq - 0.02
        if freqL >= 0.5:
            freqL = 0.5
        if freqH <= 0.0:
            freqH = 0.0

        amplitudeMS = amplitude_mono_sig_3d_lpf(fftV, freq, freqH, freqL,
                                                iu, fx, fy, fz, real_shape)
        if half_maps_given:
            amplitudeMN = amplitude_mono_sig_3d_lpf(fftN, freq, freqH, freqL,
                                                    iu, fx, fy, fz, real_shape)
            meanS, sdS2, meanN, sdN2, thr95, NS, NN = statistics_in_binary_mask(
                amplitudeMS, amplitudeMN, mask, args.significance)
            del amplitudeMN
        else:
            meanS, sdS2, meanN, sdN2, thr95, NS, NN = statistics_in_out_binary_mask(
                amplitudeMS, mask, args.significance)

        if (NS / n_voxels_original_mask) < cut_value:
            # when the 2.5% is reached then the iterative process stops
            print("Search of resolutions stopped due to mask has been completed")
            do_next_iteration = False
            resolved = resolution_map != 0
            n_voxels = int(resolved.sum().item())
            mask[~resolved] = 0
            mask[resolved] = 1
            lefttrimming = True
        else:
            if NS == 0:
                print("There are no points to compute inside the mask")
                print("If the number of computed frequencies is low, perhaps the "
                      "providedmask is not enough tight to the volume, in that "
                      "case please try another mask")
                break

            # Check local resolution
            if args.gaussian:
                threshold_noise = meanN + critical_z * _c_sqrt(sdN2)
            else:
                threshold_noise = thr95

            if meanS > max_meanS:
                max_meanS = meanS

            if meanS < 0.001 * max_meanS:
                print("Search of resolutions stopped due to too low signal")
                break

            set_local_resolution(amplitudeMS, mask, resolution_map,
                                 threshold_noise, resolution, resolution_2,
                                 half_maps_given)

            z = _c_div(meanS - meanN, _c_sqrt(sdS2 / NS + sdN2 / NN))

            if z < critical_z:
                print("Search stopped due to z>Z (hypothesis test)")
                do_next_iteration = False

            if do_next_iteration:
                if resolution < min_res:
                    do_next_iteration = False

        del amplitudeMS
        iteration += 1
        if not do_next_iteration:
            break

    if not list_res:
        raise RuntimeError("No frequency could be analysed. Check --maxRes "
                           "(the sweep starts at --maxRes and decreases towards "
                           "2*--sampling_rate) and --step.")

    if not lefttrimming:
        resolved = resolution_map != 0
        n_voxels = int(resolved.sum().item())
        mask[~resolved] = 0
        mask[resolved] = 1

    del n_voxels

    filtered_resolution = resolution_map.clone()
    post_process_local_resolutions(filtered_resolution, resolution_map, list_res,
                                   mask, sampling, args.output)


def build_parser() -> argparse.ArgumentParser:
    """CLI mirroring ``ProgMonogenicSignalRes::defineParams``."""
    p = argparse.ArgumentParser(
        prog="monores.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=("MONORES: estimates the local resolution map "
                     "from a single reconstruction or two half maps.\n"
                     "Reference: J.L. Vilas et al, MonoRes: Automatic and Accurate "
                     "Estimation of Local Resolution for Electron Microscopy Maps, "
                     "Structure, 26, 337-344, (2018)."))
    p.add_argument("--vol", default="", required=True,
                   help="Input map to estimate its local resolution map. In "
                        "half-map mode this is the first half map.")
    p.add_argument("--vol2", default="",
                   help="(Optional but recommended) Second half map.")
    p.add_argument("--mask", default="",
                   help="Mask defining the region where the protein is.")
    p.add_argument("--maskExcl", default="",
                   help="(Optional) Mask excluding a region from the estimation.")
    p.add_argument("--minRes", type=float, default=30.0,
                   help="Lowest resolution in (A). NOTE: as in the C++, this is "
                        "immediately overwritten with 2*sampling_rate and is "
                        "therefore unused.")
    p.add_argument("--maxRes", type=float, default=1.0,
                   help="Highest resolution in (A). This is the value the sweep "
                        "STARTS at and then decreases by --step until Nyquist.")
    p.add_argument("--sampling_rate", type=float, default=1.0,
                   help="Sampling rate (A/px).")
    p.add_argument("-o", "--output", default="",
                   required=True, help="Folder where the results will be stored.")
    p.add_argument("--step", type=float, default=0.25,
                   help="Resolution step in (A); clamped to >= 0.25.")
    p.add_argument("--significance", type=float, default=0.95,
                   help="Level of confidence for the signal/noise hypothesis test.")
    p.add_argument("--threads", type=int, default=4,
                   help="(Optional) Number of CPU threads (torch.set_num_threads).")
    p.add_argument("--noiseonlyinhalves", action="store_true",
                   help="The noise estimation is only performed inside the mask. "
                        "Only meaningful with two half maps.")
    p.add_argument("--gaussian", action="store_true",
                   help="Assume the noise is gaussian (faster) instead of using "
                        "the empirical distribution.")
    p.add_argument("--device", default=None,
                   help="Torch device override, e.g. cpu / cuda / cuda:1. "
                        "Defaults to cuda when available, else cpu.")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.threads and args.threads > 0:
        torch.set_num_threads(int(args.threads))

    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
