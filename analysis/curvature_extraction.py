#!/usr/bin/env python3
import json
import numpy as np
import subprocess
import os

# ============================================================
# LOCKED FILENAMES (as requested)
# ============================================================

HEX_BASELINE_FILE   = "hexagramdataset1.csv"
HEX_TUNNEL_FILE     = "tunneling_hexagram_dataset.csv"
BYTE_BASELINE_FILE  = "byte_manifold_dataset.csv"
BYTE_TUNNEL_FILE    = "tunneling_byte_manifold_dataset.csv"

# The analysis script you already have
ANALYSIS_SCRIPT = "manifold_compare_all.py"

# Temporal offsets to test
TEMPORAL_OFFSETS = [0.1, 1.0, 5.0, 30.0]

# Output file
OUTFILE = "temporal_curvature_results.json"


# ============================================================
# RUN ANALYSIS SCRIPT FOR EACH TEMPORAL OFFSET
# ============================================================

def run_analysis(dt):
    env = os.environ.copy()
    env["HEX_BASELINE_FILE"]   = HEX_BASELINE_FILE
    env["HEX_TUNNEL_FILE"]     = HEX_TUNNEL_FILE
    env["BYTE_BASELINE_FILE"]  = BYTE_BASELINE_FILE
    env["BYTE_TUNNEL_FILE"]    = BYTE_TUNNEL_FILE
    env["SAMPLING_INTERVAL"]   = str(dt)

    proc = subprocess.Popen(
        ["python", ANALYSIS_SCRIPT],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env
    )
    out, err = proc.communicate()
    return out


# ============================================================
# PARSE RESULTS FROM manifold_compare_all.py OUTPUT
# ============================================================

def parse_results(output):
    lines = output.splitlines()
    data = {}

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

    return data


# ============================================================
# TEMPORAL CURVATURE EXTRACTION
# ============================================================

def second_derivative(values, dts):
    """
    Compute temporal curvature:
        curvature(dt) = d²(values)/d(dt)²
    Using central finite differences.
    """
    vals = np.array(values)
    dts  = np.array(dts)

    curvature = []
    for i in range(1, len(vals)-1):
        # second derivative
        d2 = (vals[i+1] - 2*vals[i] + vals[i-1]) / ((dts[i+1] - dts[i])**2)
        curvature.append(float(d2))

    return curvature


# ============================================================
# MAIN
# ============================================================

def main():
    raw_results = {}

    # Run analysis for each dt
    for dt in TEMPORAL_OFFSETS:
        out = run_analysis(dt)
        parsed = parse_results(out)
        raw_results[str(dt)] = parsed

    # Extract arrays for curvature computation
    dts = TEMPORAL_OFFSETS

    hex_stat = [raw_results[str(dt)]["hex_stationary_drift"] for dt in dts]
    hex_curv = [raw_results[str(dt)]["hex_curvature_drift"] for dt in dts]
    hex_ent  = [raw_results[str(dt)]["hex_entropy_drift"] for dt in dts]

    byte_stat = [raw_results[str(dt)]["byte_stationary_drift"] for dt in dts]
    byte_curv = [raw_results[str(dt)]["byte_curvature_drift"] for dt in dts]
    byte_ent  = [raw_results[str(dt)]["byte_entropy_drift"] for dt in dts]

    # Compute temporal curvature (second derivative)
    results = {
        "hex_stationary_curvature": second_derivative(hex_stat, dts),
        "hex_curvature_curvature": second_derivative(hex_curv, dts),
        "hex_entropy_curvature": second_derivative(hex_ent, dts),

        "byte_stationary_curvature": second_derivative(byte_stat, dts),
        "byte_curvature_curvature": second_derivative(byte_curv, dts),
        "byte_entropy_curvature": second_derivative(byte_ent, dts),

        "raw": raw_results
    }

    # Save results
    with open(OUTFILE, "w") as f:
        json.dump(results, f, indent=4)

    print("\nTemporal curvature extraction complete.")
    print(f"Saved to {OUTFILE}")


if __name__ == "__main__":
    main()
