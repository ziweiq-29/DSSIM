#!/usr/bin/env python3
import argparse
import numpy as np
from skimage.metrics import structural_similarity as ssim


def _compute_dssim(orig: np.ndarray, dec: np.ndarray) -> float:
    smin = min(orig.min(), dec.min())
    smax = max(orig.max(), dec.max())
    r = smax - smin
    if r == 0:
        if smax == 0:
            sc_a1 = orig
            sc_a2 = dec
        else:
            sc_a1 = orig / smax
            sc_a2 = dec / smax
    else:
        sc_a1 = (orig - smin) / r
        sc_a2 = (dec - smin) / r
    sc_a1 = np.round(sc_a1 * 255) / 255
    sc_a2 = np.round(sc_a2 * 255) / 255

    ssim_val = ssim(
        sc_a1,
        sc_a2,
        data_range=1.0,
        gaussian_weights=True,
        sigma=1.5,
        win_size=11,
        K1=1e-4,
        K2=1e-4,
        use_sample_covariance=False,
    )
    return float(ssim_val)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--orig", required=True, help="binary float64 file for mass_orig")
    p.add_argument("--dec", required=True, help="binary float64 file for mass_dec")
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    args = p.parse_args()

    

    orig = np.fromfile(args.orig, dtype=np.float64)
    dec = np.fromfile(args.dec, dtype=np.float64)
    shape = (args.height, args.width)
    orig = orig.reshape(shape)
    dec = dec.reshape(shape)
    if orig.size == 0 or dec.size == 0 or orig.size != dec.size:
        raise SystemExit(1)
    if args.width <= 0 or args.height <= 0:
        raise SystemExit(1)
    if orig.size != args.width * args.height:
        raise SystemExit(1)

    print(_compute_dssim(orig, dec))


if __name__ == "__main__":
    main()