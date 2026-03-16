
#!/usr/bin/env python3
"""
LibPressio external metric: compute DSSIM between original (-i) and decompressed (-W) data.

This script is intended to be launched via libpressio's external metric plugin, which
passes the following CLI arguments:
  --input         path to the temporary original (uncompressed) data
  --decompressed  path to the temporary decompressed data (corresponding to -W)
  --dim           can appear multiple times to specify each dimension
  --type          data type string matching pressio's -t (default: float32)

The script reads both arrays, reshapes according to --dim if provided, normalizes them
to [0, 1], computes SSIM, then outputs DSSIM = (1 - SSIM) / 2 via the JSON external API:

  external:api=json:1
  {"dssim": <value>}
"""

import argparse
import json
import os
import sys
from typing import List
from skimage.metrics import structural_similarity as ssim


# Avoid picking up libpressio-env's Python packages (numpy for py3.11 etc.)
os.environ.pop("PYTHONPATH", None)
_pp = "/anvil/projects/x-cis240669/libpressio-env"
sys.path = [p for p in sys.path if not (p.startswith(_pp) or _pp in p)]

import numpy as np


def _dtype_from_str(t: str):
    t = (t or "").lower()
    mapping = {
        "f": np.float32,
        "float": np.float32,
        "float32": np.float32,
        "f32": np.float32,
        "double": np.float64,
        "float64": np.float64,
        "f64": np.float64,
    }
    return mapping.get(t, np.float32)


def _load_array(path: str, dtype, dims: List[int]) -> np.ndarray:
    arr = np.fromfile(path, dtype=dtype)
    if dims:
        expected = int(np.prod(dims))
        if arr.size == expected:
            arr = arr.reshape(tuple(dims))
    return arr

def compute_metrics(orig: np.ndarray, dec: np.ndarray):
    """Build vector metrics consumed by downstream QOI scripts.

    For DSSIM/CESM raw fields, we use element-wise pairing:
    - dists: absolute difference between original and decompressed samples
    - mass_orig: original sample values
    - mass_dec: decompressed sample values
    """
    mass_orig = np.asarray(orig, dtype=np.float64)
    mass_dec = np.asarray(dec, dtype=np.float64)
    dists = np.abs(mass_orig - mass_dec).ravel()
    return dists, mass_orig, mass_dec



def _output_default(reason: str = "") -> None:
    if reason:
        print(f"pressio_dssim: {reason}", file=sys.stderr)
    print("external:api=json:1", flush=True)
    print(json.dumps({"dssim": None}), flush=True)


def main(argv: List[str] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--api", type=int, default=None, help="external API version (libpressio)")
    p.add_argument("--input", default=None, help="path to original/uncompressed temp file")
    p.add_argument("--decompressed", default=None, help="path to decompressed temp file")
    p.add_argument("--dim", type=int, action="append", default=[], help="dimensions (repeat)")
    p.add_argument("--type", default="float32", help="data type, matches pressio -t")
    args, _ = p.parse_known_args(argv)

    # Debug: log how external was invoked and what files we see
    try:
        print(
            f"pressio_dssim: argv={sys.argv}, input={args.input}, "
            f"decompressed={args.decompressed}, dim={args.dim}, type={args.type}",
            file=sys.stderr,
        )
        if args.input:
            print(
                f"pressio_dssim: input_exists={os.path.isfile(args.input)} "
                f"size={os.path.getsize(args.input) if os.path.isfile(args.input) else 'NA'}",
                file=sys.stderr,
            )
        if args.decompressed:
            print(
                f"pressio_dssim: dec_exists={os.path.isfile(args.decompressed)} "
                f"size={os.path.getsize(args.decompressed) if os.path.isfile(args.decompressed) else 'NA'}",
                file=sys.stderr,
            )
    except Exception:
        # Debug logging failure should not break metric
        pass

    if not args.input or not args.decompressed:
        _output_default("missing --input or --decompressed")
        return 0
    if not os.path.isfile(args.input) or not os.path.isfile(args.decompressed):
        _output_default(
            f"file missing: input_exists={os.path.isfile(args.input)}, "
            f"dec_exists={os.path.isfile(args.decompressed)}"
        )
        return 0

    try:
        dtype = _dtype_from_str(args.type)
        dims = [int(d) for d in args.dim] if args.dim else []

        orig = _load_array(args.input, dtype, dims)
        dec = _load_array(args.decompressed, dtype, dims)
        #改成mass_dist 那些

        if orig.size == 0 or dec.size == 0:
            _output_default(f"empty arrays: orig.size={orig.size}, dec.size={dec.size}")
            return 0
        if orig.size != dec.size:
            _output_default(
                f"size mismatch: orig.size={orig.size}, dec.size={dec.size}"
            )
            return 0

        dists, mass_orig, mass_dec = compute_metrics(orig, dec)

        np.save("dists.npy", dists)
        np.save("mass_orig.npy", mass_orig)
        np.save("mass_dec.npy", mass_dec)   
        return 0

        
    except Exception as e:
        print(f"pressio_dssim: failed to compute dssim: {e}", file=sys.stderr)
        _output_default()
        return 1


if __name__ == "__main__":
    sys.exit(main())
