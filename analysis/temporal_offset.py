#!/usr/bin/env python3
import os
import json
import subprocess

# ============================================================
# CONFIG
# ============================================================

# Your existing analysis script
ANALYSIS_SCRIPT = "manifold_compare_all.py"

# Temporal regimes to test
TEMPORAL_OFFSETS = [
    "0.1",
    "1.0",
    "5.0",
    "30.0"
]

# Dataset directory structure
HEX_BASE_TEMPLATE   = "hexagramdataset1.csv"
HEX_TUNNEL_TEMPLATE = "tunneling_hexagram_dataset.csv"

BYTE_BASE_TEMPLATE   = "byte_manifold_dataset.csv"
BYTE_TUNNEL_TEMPLATE = "tunneling_byte_manifold_dataset.csv"

# Output file
RESULTS_FILE = "temporal_offset_results.json"


# ============================================================
# RUN ANALYSIS SCRIPT FOR EACH TEMPORAL OFFSET
# ============================================================

def run_analysis(dt):
    print(f"\n=== Running temporal offset dt={dt} ===")

    # Build filenames
    hex_base   = HEX_BASE_TEMPLATE.format(dt=dt)
    hex_tunnel = HEX_TUNNEL_TEMPLATE.format(dt=dt)
    byte_base  = BYTE_BASE_TEMPLATE.format(dt=dt)
    byte_tunnel= BYTE_TUNNEL_TEMPLATE.format(dt=dt)

    # Ensure files exist
    for f in [hex_base, hex_tunnel, byte_base, byte_tunnel]:
        if not os.path.exists(f):
            print(f"[WARN] Missing dataset: {f}")
            return None

    # Run your existing script with environment overrides
    env = os.environ.copy()
    env["HEX_BASELINE_FILE"]   = hex_base
    env["HEX_TUNNEL_FILE"]     = hex_tunnel
    env["BYTE_BASELINE_FILE"]  = byte_base
    env["BYTE_TUNNEL_FILE"]    = byte_tunnel

    # Capture output
    proc = subprocess.Popen(
        ["python", ANALYSIS_SCRIPT],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True
    )
    out, err = proc.communicate()

    if err.strip():
        print("[ERROR]", err)

    return out


# ============================================================
# PARSE RESULTS FROM YOUR EXISTING SCRIPT OUTPUT
# ============================================================

def parse_results(output):
    lines = output.splitlines()
    data = {}

    # Drift / curvature / entropy
    for line in lines:
        if "hexagram:" in line:
            parts = line.split()
            data["hex_stationary_drift"] = float(parts[1].split("=")[1])
            data["hex_curvature_drift"]  = float(parts[2].split("=")[1])
            data["hex_entropy_drift"]    = float(parts[3].split("=")[1])
        if "byte:" in line:
            parts = line.split()
            data["byte_stationary_drift"] = float(parts[1].split("=")[1])
            data["byte_curvature_drift"]  = float(parts[2].split("=")[1])
            data["byte_entropy_drift"]    = float(parts[3].split("=")[1])

        # Basin persistence
        if "hexagram manifold:" in line:
            pass
        if "baseline top-10 clusters:" in line and "hex" in data:
            pass

        if "overlap_fraction" in line:
            frac = float(line.split("=")[-1])
            if "hex_overlap_fraction" not in data:
                data["hex_overlap_fraction"] = frac
            else:
                data["byte_overlap_fraction"] = frac

    return data


# ============================================================
# MAIN
# ============================================================

def main():
    results = {}

    for dt in TEMPORAL_OFFSETS:
        out = run_analysis(dt)
        if out is None:
            continue

        parsed = parse_results(out)
        results[dt] = parsed

    # Save results
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4)

    print("\n=== Temporal offset testing complete ===")
    print(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
