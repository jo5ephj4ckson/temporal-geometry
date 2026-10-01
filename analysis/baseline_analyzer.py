import csv
from collections import Counter, defaultdict
from datetime import datetime

DATA_FILE = "hexagramdataset1.csv"  # or your actual filename

def parse_hexagram_lines(s):
    return [int(x) for x in s.split("-")]

def parse_changing_lines(s):
    if s == "none":
        return []
    return [int(x) for x in s.split("-")]

def load_rows():
    rows = []
    with open(DATA_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def analyze_hexagrams(rows):
    hexagram_counter = Counter()
    changing_count = Counter()
    line_value_counter = Counter()
    changing_line_positions = Counter()

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

    return {
        "hexagram_counter": hexagram_counter,
        "changing_count": changing_count,
        "line_value_counter": line_value_counter,
        "changing_line_positions": changing_line_positions,
    }

def analyze_system_state(rows):
    cpu_series = []
    ram_series = []
    disk_series = []
    time_series = []

    for row in rows:
        ts = float(row["timestamp_unix"])
        cpu = float(row["cpu_percent"])
        ram = float(row["ram_percent"])
        disk = float(row["disk_percent"])
        time_series.append(ts)
        cpu_series.append(cpu)
        ram_series.append(ram)
        disk_series.append(disk)

    def basic_stats(series):
        if not series:
            return {}
        n = len(series)
        avg = sum(series) / n
        mn = min(series)
        mx = max(series)
        return {"count": n, "avg": avg, "min": mn, "max": mx}

    return {
        "cpu": basic_stats(cpu_series),
        "ram": basic_stats(ram_series),
        "disk": basic_stats(disk_series),
    }

def correlate_hexagrams_with_cpu(rows):
    # average CPU per hexagram pattern
    cpu_by_hex = defaultdict(list)
    for row in rows:
        hex_str = row["hexagram_lines"]
        cpu = float(row["cpu_percent"])
        cpu_by_hex[hex_str].append(cpu)

    avg_cpu_by_hex = {}
    for hex_str, vals in cpu_by_hex.items():
        avg_cpu_by_hex[hex_str] = sum(vals) / len(vals)
    return avg_cpu_by_hex

def main():
    print("Loading dataset...")
    rows = load_rows()
    print(f"Loaded {len(rows)} samples")

    print("\n=== HEXAGRAM STRUCTURE ANALYSIS ===")
    hstats = analyze_hexagrams(rows)

    print("\nTop 10 hexagrams by frequency:")
    for hex_str, count in hstats["hexagram_counter"].most_common(10):
        print(f"  {hex_str}: {count}")

    print("\nLine value distribution (6/7/8/9):")
    for v in sorted(hstats["line_value_counter"].keys()):
        print(f"  {v}: {hstats['line_value_counter'][v]}")

    print("\nChanging line count distribution:")
    for k in sorted(hstats["changing_count"].keys()):
        print(f"  {k} changing lines: {hstats['changing_count'][k]} samples")

    print("\nChanging line position frequency (1–6):")
    for pos in range(1, 7):
        print(f"  line {pos}: {hstats['changing_line_positions'][pos]}")

    print("\n=== SYSTEM STATE ANALYSIS ===")
    sstats = analyze_system_state(rows)
    for key in ["cpu", "ram", "disk"]:
        st = sstats[key]
        print(f"{key.upper()}: count={st['count']} avg={st['avg']:.2f} min={st['min']:.2f} max={st['max']:.2f}")

    print("\n=== HEXAGRAM ↔ CPU CORRELATION (average CPU per hexagram) ===")
    cpu_corr = correlate_hexagrams_with_cpu(rows)
    # show top 10 by average CPU
    for hex_str, avg_cpu in sorted(cpu_corr.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {hex_str}: avg CPU {avg_cpu:.2f}")

if __name__ == "__main__":
    main()
