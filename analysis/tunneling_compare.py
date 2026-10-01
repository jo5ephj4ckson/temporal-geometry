import csv
from collections import Counter, defaultdict

BASELINE_FILE = "hexagramdataset1.csv"
TUNNEL_FILE   = "tunneling_hexagram_dataset.csv"

def parse_hexagram_lines(s):
    return [int(x) for x in s.split("-")]

def parse_changing_lines(s):
    if s == "none":
        return []
    return [int(x) for x in s.split("-")]

def load_rows(filename):
    rows = []
    with open(filename, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def analyze_dataset(rows, include_cpu=False):
    hexagram_counter = Counter()
    changing_count = Counter()
    line_value_counter = Counter()
    changing_line_positions = Counter()
    cpu_values = []

    for row in rows:
        hex_str = row["hexagram_lines"]
        chg_str = row["changing_lines"]

        hexagram_counter[hex_str] += 1

        lines = parse_hexagram_lines(hex_str)
        for i, v in enumerate(lines, start=1):
            line_value_counter[v] += 1

        changing = parse_changing_lines(chg_str)
        for pos in changing:
            changing_line_positions[pos] += 1
        changing_count[len(changing)] += 1

        if include_cpu:
            cpu_values.append(float(row["cpu_percent"]))

    stats = {
        "hexagram_counter": hexagram_counter,
        "changing_count": changing_count,
        "line_value_counter": line_value_counter,
        "changing_line_positions": changing_line_positions,
    }

    if include_cpu:
        stats["cpu_avg"] = sum(cpu_values) / len(cpu_values)
        stats["cpu_min"] = min(cpu_values)
        stats["cpu_max"] = max(cpu_values)

    return stats

def print_top_hexagrams(label, counter, n=10):
    print(f"\nTop {n} hexagrams in {label}:")
    for hex_str, count in counter.most_common(n):
        print(f"  {hex_str}: {count}")

def print_distribution(label, counter):
    print(f"\nLine value distribution for {label}:")
    for v in sorted(counter.keys()):
        print(f"  {v}: {counter[v]}")

def print_changing_distribution(label, counter):
    print(f"\nChanging line count distribution for {label}:")
    for k in sorted(counter.keys()):
        print(f"  {k} changing lines: {counter[k]} samples")

def print_position_distribution(label, counter):
    print(f"\nChanging line position frequency for {label}:")
    for pos in range(1, 7):
        print(f"  line {pos}: {counter[pos]}")

def compare_distributions(base, tunnel):
    print("\n=== LINE VALUE DELTA (tunneling - baseline) ===")
    for v in sorted(base["line_value_counter"].keys()):
        b = base["line_value_counter"][v]
        t = tunnel["line_value_counter"][v]
        print(f"  {v}: baseline={b}, tunneling={t}, delta={t - b}")

    print("\n=== CHANGING LINE COUNT DELTA ===")
    for k in sorted(base["changing_count"].keys()):
        b = base["changing_count"][k]
        t = tunnel["changing_count"][k]
        print(f"  {k} changing lines: baseline={b}, tunneling={t}, delta={t - b}")

    print("\n=== POSITION DELTA ===")
    for pos in range(1, 7):
        b = base["changing_line_positions"][pos]
        t = tunnel["changing_line_positions"][pos]
        print(f"  line {pos}: baseline={b}, tunneling={t}, delta={t - b}")

def main():
    print("Loading datasets...")
    baseline_rows = load_rows(BASELINE_FILE)
    tunnel_rows   = load_rows(TUNNEL_FILE)

    print(f"Baseline samples: {len(baseline_rows)}")
    print(f"Tunneling samples: {len(tunnel_rows)}")

    print("\nAnalyzing baseline RNG dataset...")
    baseline_stats = analyze_dataset(baseline_rows, include_cpu=False)

    print("\nAnalyzing tunneling entropy dataset...")
    tunnel_stats = analyze_dataset(tunnel_rows, include_cpu=True)

    print_top_hexagrams("baseline RNG", baseline_stats["hexagram_counter"])
    print_top_hexagrams("tunneling entropy", tunnel_stats["hexagram_counter"])

    print_distribution("baseline RNG", baseline_stats["line_value_counter"])
    print_distribution("tunneling entropy", tunnel_stats["line_value_counter"])

    print_changing_distribution("baseline RNG", baseline_stats["changing_count"])
    print_changing_distribution("tunneling entropy", tunnel_stats["changing_count"])

    print_position_distribution("baseline RNG", baseline_stats["changing_line_positions"])
    print_position_distribution("tunneling entropy", tunnel_stats["changing_line_positions"])

    print("\n=== CPU STATS FOR TUNNELING DATASET ===")
    print(f"  avg CPU: {tunnel_stats['cpu_avg']:.2f}")
    print(f"  min CPU: {tunnel_stats['cpu_min']:.2f}")
    print(f"  max CPU: {tunnel_stats['cpu_max']:.2f}")

    print("\n=== COMPARISON DELTAS ===")
    compare_distributions(baseline_stats, tunnel_stats)

if __name__ == "__main__":
    main()
