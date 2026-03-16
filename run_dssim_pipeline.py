#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys

# Keep this process isolated from libpressio-env Python path pollution.
os.environ.pop("PYTHONPATH", None)
_pp = "/anvil/projects/x-cis240669/libpressio-env"
sys.path = [p for p in sys.path if not (p.startswith(_pp) or _pp in p)]

import numpy as np


def _print_empty_metrics(exit_code: int = 0) -> int:
    print("external:api=json:1", flush=True)
    print(json.dumps({"dists": [], "mass_orig": [], "mass_dec": []}), flush=True)
    return exit_code


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="DSSIM external pipeline: delegate to pressio_dssim.py and emit JSON metrics"
    )
    parser.add_argument("--external_mode", action="store_true")
    parser.add_argument("--input", help="Input path from libpressio external plugin")
    parser.add_argument("--decompressed", help="Decompressed path from libpressio external plugin")
    parser.add_argument("--dim", type=int, action="append", help="Dimensions (repeat --dim)")
    parser.add_argument("--type", default="float32", help="Data type (matches pressio -t)")
    parser.add_argument(
        "--external_script",
        default="/anvil/projects/x-cis240669/DSSIM/pressio_dssim.py",
        help="Path to metric script that writes dists.npy/mass_orig.npy/mass_dec.npy",
    )
    args, _ = parser.parse_known_args(argv)

    dims = args.dim or []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    external_script = (
        args.external_script
        if os.path.isabs(args.external_script)
        else os.path.join(script_dir, args.external_script)
    )

    # Probe call from libpressio has no --decompressed; respond with valid empty JSON payload.
    if not args.decompressed:
        return _print_empty_metrics(0)

    if not args.input:
        print("[run_dssim_pipeline] missing --input", file=sys.stderr)
        return _print_empty_metrics(1)

    if not os.path.exists(external_script):
        print(f"[run_dssim_pipeline] missing external script: {external_script}", file=sys.stderr)
        return _print_empty_metrics(1)

    env = os.environ.copy()
    cmd = [
        sys.executable,
        external_script,
        "--input",
        args.input,
        "--decompressed",
        args.decompressed,
        "--type",
        args.type,
    ]
    for d in dims:
        cmd.extend(["--dim", str(d)])

    out_dir = os.path.dirname(os.path.abspath(external_script))
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        env=env,
        cwd=out_dir,
    )
    if result.stderr:
        sys.stderr.write(result.stderr)
    if result.returncode != 0:
        print(
            f"[run_dssim_pipeline] external script failed with code {result.returncode}",
            file=sys.stderr,
        )
        return _print_empty_metrics(result.returncode)

    dists_path = os.path.join(out_dir, "dists.npy")
    mass_orig_path = os.path.join(out_dir, "mass_orig.npy")
    mass_dec_path = os.path.join(out_dir, "mass_dec.npy")
    if not (os.path.exists(dists_path) and os.path.exists(mass_orig_path) and os.path.exists(mass_dec_path)):
        print(f"[run_dssim_pipeline] missing .npy artifacts in {out_dir}", file=sys.stderr)
        return _print_empty_metrics(1)

    dists = np.load(dists_path)
    mass_orig = np.load(mass_orig_path)
    mass_dec = np.load(mass_dec_path)

    print("external:api=json:1", file=sys.stdout, flush=True)
    print(
        json.dumps(
            {
                "dists": dists.tolist(),
                "mass_orig": mass_orig.tolist(),
                "mass_dec": mass_dec.tolist(),
            }
        ),
        file=sys.stdout,flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
