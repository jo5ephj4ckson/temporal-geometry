import sys
import math
from datetime import datetime

# Change this if your file is somewhere else
INPUT_FILE = "raw_temporal_entropy.txt"

def parse_line(line):
    """Parse a single entropy line: timestamp value"""
    parts = line.strip().split()
    if len(parts) != 2:
        return None, None
    ts = datetime.fromisoformat(parts[0].replace("Z", "+00:00"))
    val = float(parts[1])
    return ts, val

def main():
    timestamps = []
    values = []

    print(f"Loading: {INPUT_FILE}")

    with open(INPUT_FILE, "r") as f:
        for line in f:
            ts, val = parse_line(line)
            if ts is not None:
                timestamps.append(ts)
                values.append(val)

    if not values:
        print("No valid entropy data found.")
        sys.exit(1)

    # Basic stats
    min_val = min(values)
    max_val = max(values)
    mean_val = sum(values) / len(values)
    std_val = math.sqrt(sum((v - mean_val)**2 for v in values) / len(values))

    # Drift analysis
    drift = values[-1] - values[0]

    # Timestamp spacing
    if len(timestamps) > 1:
        deltas = [(timestamps[i+1] - timestamps[i]).total_seconds()
                  for i in range(len(timestamps)-1)]
        avg_spacing = sum(deltas) / len(deltas)
    else:
        avg_spacing = 0

    print("\n=== RAW TEMPORAL ENTROPY ANALYSIS ===")
    print(f"Total samples: {len(values)}")
    print(f"Min entropy: {min_val}")
    print(f"Max entropy: {max_val}")
    print(f"Mean entropy: {mean_val}")
    print(f"Std deviation: {std_val}")
    print(f"Drift (last - first): {drift}")
    print(f"Average timestamp spacing (sec): {avg_spacing}")

    # Optional CSV export
    with open("entropy_export.csv", "w") as out:
        out.write("timestamp,entropy\n")
        for ts, val in zip(timestamps, values):
            out.write(f"{ts.isoformat()},{val}\n")

    print("\nCSV export written to entropy_export.csv")

if __name__ == "__main__":
    main()
