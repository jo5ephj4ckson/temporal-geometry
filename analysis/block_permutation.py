import numpy as np
import pandas as pd
import umap
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler


def run_temporal_probe_scrubbed_pipeline(
    csv_filepath, n_clusters=8, n_permutations=1000, block_size=100
):
  print("--- Step 1: Loading & Inspecting Dataset ---")
  df = pd.read_csv(csv_filepath)
  print(f"Dataset Shape: {df.shape}")

  # 1. Purge metadata, text strings, and direct/indirect time features
  cols_to_drop = [
      'timestamp_iso',
      'timestamp_unix',
      'system_boot_time_unix',
      'process_start_unix',
      'time_since_process_start_seconds',
      'timestamp',
      'time',
      'notes',
      'machine_name',
      'os_info',
      'python_version',
      'script_name',
      'working_dir',
      'sampling_interval_seconds',
      'index',
      'Unnamed: 0',
  ]

  df_clean = df.drop(
      columns=[col for col in cols_to_drop if col in df.columns]
  )

  # 2. Parse hexagram binary representations
  # If 'hexagram_lines' is a string like "101101" or a list string "[1, 0, 1, 1, 0, 1]", parse it:
  if 'hexagram_lines' in df_clean.columns:
    # Convert string representation to individual binary columns
    lines_series = df_clean['hexagram_lines'].astype(str)

    # Handle string digits or array representations
    parsed_lines = []
    for val in lines_series:
      digits = [int(char) for char in val if char in '01']
      # Ensure 6 bits per hexagram
      if len(digits) == 6:
        parsed_lines.append(digits)
      else:
        parsed_lines.append([0, 0, 0, 0, 0, 0])

    hex_df = pd.DataFrame(
        parsed_lines,
        columns=[
            'line1',
            'line2',
            'line3',
            'line4',
            'line5',
            'line6',
        ],
    )
    df_clean = pd.concat(
        [df_clean.drop(columns=['hexagram_lines']), hex_df], axis=1
    )

  if 'changing_lines' in df_clean.columns:
    df_clean = df_clean.drop(columns=['changing_lines'])

  # 3. Separate binary hexagram features from numeric system metrics
  hexagram_cols = ['line1', 'line2', 'line3', 'line4', 'line5', 'line6']
  continuous_cols = [
      col
      for col in df_clean.columns
      if col not in hexagram_cols
      and pd.api.types.is_numeric_dtype(df_clean[col])
  ]

  print(f"Hexagram Binary Columns ({len(hexagram_cols)}): {hexagram_cols}")
  print(
      f"Continuous System Numeric Columns ({len(continuous_cols)}):"
      f" {continuous_cols}"
  )

  print("\n--- Step 2: Stationary Differencing & Feature Scaling ---")
  df_processed = df_clean.copy()

  # Apply first-order differencing ONLY to continuous numeric metrics
  if continuous_cols:
    df_processed[continuous_cols] = df_processed[continuous_cols].diff()

  # Drop NaN row resulting from .diff()
  df_processed = df_processed.dropna().reset_index(drop=True)

  # Apply Z-score standardization across continuous features
  scaler = StandardScaler()
  if continuous_cols:
    df_processed[continuous_cols] = scaler.fit_transform(
        df_processed[continuous_cols]
    )

  feature_cols = hexagram_cols + continuous_cols
  X = df_processed[feature_cols].values
  print(f"Cleaned & Scaled Feature Matrix X shape: {X.shape}")

  print("\n--- Step 3: Re-Running K-Means & UMAP Projections ---")
  kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
  clusters = kmeans.fit_predict(X)
  df_processed['kmeans_cluster'] = clusters

  reducer = umap.UMAP(
      n_components=2, random_state=42, n_neighbors=30, min_dist=0.1
  )
  umap_embedding = reducer.fit_transform(X)
  df_processed['umap_1'] = umap_embedding[:, 0]
  df_processed['umap_2'] = umap_embedding[:, 1]

  print("UMAP Projection complete.")

  print("\n--- Step 4: Block Permutation Test ---")
  hexagram_matrix = df_processed[hexagram_cols].values
  system_matrix = (
      df_processed[continuous_cols].values if continuous_cols else X
  )

  system_kmeans = KMeans(
      n_clusters=4, random_state=42, n_init=10
  ).fit_predict(system_matrix)

  # Pack 6 binary lines into integer state representations (0 to 63)
  powers_of_two = 2 ** np.arange(6)
  hexagram_states = np.dot(hexagram_matrix.astype(int), powers_of_two)

  observed_nmi = normalized_mutual_info_score(system_kmeans, hexagram_states)
  print(
      f"Observed Clean Normalized Mutual Information (NMI): {observed_nmi:.6f}"
  )

  n_samples = len(hexagram_matrix)
  n_blocks = n_samples // block_size
  permuted_nmis = []

  print(
      f"Running {n_permutations} trials with block size = {block_size}"
      " samples..."
  )
  for i in range(n_permutations):
    block_indices = np.arange(n_blocks)
    np.random.shuffle(block_indices)

    shuffled_hexagrams = []
    for idx in block_indices:
      shuffled_hexagrams.append(
          hexagram_states[idx * block_size : (idx + 1) * block_size]
      )

    remainder = n_samples % block_size
    if remainder > 0:
      shuffled_hexagrams.append(hexagram_states[-remainder:])

    permuted_states = np.concatenate(shuffled_hexagrams)
    p_nmi = normalized_mutual_info_score(system_kmeans, permuted_states)
    permuted_nmis.append(p_nmi)

  permuted_nmis = np.array(permuted_nmis)
  p_value = np.sum(permuted_nmis >= observed_nmi) / float(n_permutations)

  print("\n================ FINAL RESULTS ================")
  print(f"Observed NMI:               {observed_nmi:.6f}")
  print(f"Mean Permuted NMI (Null):   {np.mean(permuted_nmis):.6f}")
  print(f"Max Permuted NMI (Null):    {np.max(permuted_nmis):.6f}")
  print(f"Block Permutation P-value:  {p_value:.6f}")
  print("===============================================")

  if p_value < 0.05:
    print(
        "VERDICT: Statistically significant coupling detected after scrubbing"
        " temporal leakage."
    )
  else:
    print(
        "VERDICT: Null hypothesis confirmed. No coupling detected beyond random"
        " chance."
    )

  return df_processed, umap_embedding, p_value


if __name__ == "__main__":
  csv_filename = "temporal_probe_dataset.csv"  # Ensure this matches your exact CSV filename
  df_scrubbed, umap_coords, p_val = run_temporal_probe_scrubbed_pipeline(
      csv_filepath=csv_filename,
      n_clusters=8,
      n_permutations=1000,
      block_size=100,
  )