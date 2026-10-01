import csv

def parse_hexagram_lines(s):
    # s looks like "7-8-7-7-7-6"
    parts = s.split('-')
    return [int(p) for p in parts]

def main(input_path, output_path):
    with open(input_path, newline='', encoding='utf-8') as infile, \
         open(output_path, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.DictReader(infile)

        fieldnames = [
            'timestamp_iso',
            'timestamp_unix',
            'line1', 'line2', 'line3', 'line4', 'line5', 'line6',
            'changing_lines',
            'cpu_percent',
            'ram_percent',
            'disk_percent',
            'time_since_process_start_seconds',
            'system_boot_time_unix'
        ]

        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            # Parse the hexagram lines
            lines = parse_hexagram_lines(row['hexagram_lines'])

            out_row = {
                'timestamp_iso': row['timestamp_iso'],
                'timestamp_unix': row['timestamp_unix'],
                'line1': lines[0],
                'line2': lines[1],
                'line3': lines[2],
                'line4': lines[3],
                'line5': lines[4],
                'line6': lines[5],
                'changing_lines': row['changing_lines'],
                'cpu_percent': row['cpu_percent'],
                'ram_percent': row['ram_percent'],
                'disk_percent': row['disk_percent'],
                'time_since_process_start_seconds': row['time_since_process_start_seconds'],
                'system_boot_time_unix': row['system_boot_time_unix']
            }

            writer.writerow(out_row)

    print("Structured dataset saved as:", output_path)

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print("Usage: python probe_structuring.py input.csv output.csv")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
