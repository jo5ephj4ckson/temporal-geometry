# Analysis Scripts Manifest

Auto-generated index of code structure, functions, and intent.

## `analysis\16bit_curve.py`

**Overview:** No module docstring

**Key Imports:** `Counter, csv, defaultdict, math`

*No functions defined (script may be procedural/linear execution).*

---

## `analysis\16bit_shuffle.py`

**Overview:** No module docstring

**Key Imports:** `Counter, csv, defaultdict, random`

*No functions defined (script may be procedural/linear execution).*

---

## `analysis\16bit_synthetic.py`

**Overview:** No module docstring

**Key Imports:** `Counter, csv, defaultdict, random`

*No functions defined (script may be procedural/linear execution).*

---

## `analysis\16bit_tunneling.py`

**Overview:** No module docstring

**Key Imports:** `UTC, csv, datetime, hashlib, os, platform, psutil, sounddevice, time`

**Functions:**
- `get_entropy_block()`: Capture raw audio bytes from LED tunneling probe.
- `entropy_to_uint16()`: Hash entropy and return a 16-bit integer (0..65535).
- `init_dataset()`: No description
- `record_symbol()`: No description

---

## `analysis\alt_clustering_analysis.py`

**Overview:** No module docstring

**Key Imports:** `DBSCAN, KMeans, MiniBatchKMeans, numpy, pandas`

**Functions:**
- `build_markov_chain()`: No description
- `print_attractors()`: No description
- `drift()`: No description

---

## `analysis\analyze_clusters.py`

**Overview:** No module docstring

**Key Imports:** `KMeans, numpy, pandas`

**Functions:**
- `load_hexagrams()`: No description
- `build_chain()`: No description
- `analyze_growth()`: No description

---

## `analysis\analyze_entropy.py`

**Overview:** No module docstring

**Key Imports:** `datetime, math, sys`

**Functions:**
- `parse_line()`: Parse a single entropy line: timestamp value
- `main()`: No description

---

## `analysis\android_analyzer.py`

**Overview:** No module docstring

**Key Imports:** `AgglomerativeClustering, KMeans, numpy, pandas`

*No functions defined (script may be procedural/linear execution).*

---

## `analysis\attractor_basin.py`

**Overview:** No module docstring

**Key Imports:** `KMeans, numpy, pandas`

**Functions:**
- `expand_hexagram_lines()`: No description

---

## `analysis\attractor_visualization.py`

**Overview:** No module docstring

**Key Imports:** `Axes3D, matplotlib.pyplot, numpy, pandas`

**Functions:**
- `load_points()`: No description
- `plot_2d()`: No description
- `plot_3d()`: No description
- `main()`: No description

---

## `analysis\audio_diagnostic.py`

**Overview:** No module docstring

**Key Imports:** `numpy, sounddevice, time`

*No functions defined (script may be procedural/linear execution).*

---

## `analysis\baseline_analyzer.py`

**Overview:** No module docstring

**Key Imports:** `Counter, csv, datetime, defaultdict`

**Functions:**
- `parse_hexagram_lines()`: No description
- `parse_changing_lines()`: No description
- `load_rows()`: No description
- `analyze_hexagrams()`: No description
- `analyze_system_state()`: No description
- `correlate_hexagrams_with_cpu()`: No description
- `main()`: No description
- `basic_stats()`: No description

---

## `analysis\basin_fingerprinting.py`

**Overview:** No module docstring

**Key Imports:** `KMeans, cdist, numpy, pandas`

**Functions:**
- `load_hexagram_dataset()`: No description
- `cluster_and_centroids()`: No description
- `fingerprint()`: No description

---

## `analysis\basin_stability.py`

**Overview:** No module docstring

**Key Imports:** `KMeans, numpy, pandas`

**Functions:**
- `load_and_expand()`: No description
- `build_chain()`: No description

---

## `analysis\block_permutation.py`

**Overview:** No module docstring

**Key Imports:** `KMeans, StandardScaler, chi2_contingency, normalized_mutual_info_score, numpy, pandas, umap`

**Functions:**
- `run_temporal_probe_scrubbed_pipeline()`: No description

---

## `analysis\build_matrix.py`

**Overview:** No module docstring

**Key Imports:** `numpy, pandas`

*No functions defined (script may be procedural/linear execution).*

---

## `analysis\byte_manifold_analyze.py`

**Overview:** No module docstring

**Key Imports:** `KMeans, numpy, pandas`

**Functions:**
- `load_byte_dataset()`: No description
- `build_markov()`: No description
- `drift()`: No description

---

## `analysis\byte_manifold_clusters.py`

**Overview:** No module docstring

**Key Imports:** `KMeans, numpy, pandas`

**Functions:**
- `shannon_entropy()`: No description
- `load_hex_dataset()`: No description
- `load_byte_dataset()`: No description
- `cluster_and_markov()`: No description
- `distribution_drift()`: No description
- `basin_persistence()`: No description
- `main()`: No description

---

## `analysis\byte_manifold_compare.py`

**Overview:** No module docstring

**Key Imports:** `Counter, csv, os`

**Functions:**
- `load_hexagram_change_counts()`: Extract changing-line counts from hexagram CSV.
- `load_byte_change_counts()`: Extract bit-change counts from byte-manifold CSV.
- `curvature_from_counts()`: Compute normalized curvature C_norm.
- `main()`: No description

---

## `analysis\centroid_heatmap.py`

**Overview:** No module docstring

**Key Imports:** `matplotlib.pyplot, numpy, pandas`

*No functions defined (script may be procedural/linear execution).*

---

## `analysis\cluster_comparison.py`

**Overview:** No module docstring

**Key Imports:** `KMeans, numpy, pandas`

**Functions:**
- `expand_hexagram_lines()`: No description
- `build_markov()`: No description
- `drift()`: No description

---

## `analysis\cluster_permutation.py`

**Overview:** No module docstring

**Key Imports:** `KMeans, numpy, pandas`

**Functions:**
- `load_and_expand()`: No description
- `build_chain()`: No description

---

## `analysis\cluster_similarity.py`

**Overview:** No module docstring

**Key Imports:** `KMeans, numpy, pandas`

**Functions:**
- `clean_and_parse()`: No description
- `build_markov()`: No description
- `drift()`: No description

---

## `analysis\cluster_structure.py`

**Overview:** No module docstring

**Key Imports:** `hdbscan, numpy, pandas`

**Functions:**
- `parse_changing()`: No description

---

## `analysis\color_umap.py`

**Overview:** No module docstring

**Key Imports:** `StandardScaler, hdbscan, matplotlib.pyplot, numpy, pandas, umap`

*No functions defined (script may be procedural/linear execution).*

---

## `analysis\columnscheck.py`

**Overview:** No module docstring

**Key Imports:** `pandas`

*No functions defined (script may be procedural/linear execution).*

---

## `analysis\cross_compare.py`

**Overview:** No module docstring

**Key Imports:** `KMeans, numpy, pairwise_distances, pandas`

**Functions:**
- `parse_hexagram_string()`: No description
- `extract_android_vectors()`: No description
- `extract_hexagram_vectors()`: No description
- `run_kmeans()`: No description
- `cluster_counts()`: No description
- `transition_matrix()`: No description
- `center_divergence()`: No description
- `normalize()`: No description
- `kl_divergence()`: No description

---

## `analysis\curvature_extraction.py`

**Overview:** No module docstring

**Key Imports:** `json, numpy, os, subprocess`

**Functions:**
- `run_analysis()`: No description
- `parse_results()`: No description
- `second_derivative()`: Compute temporal curvature:
- `main()`: No description

---

## `analysis\cycle_detection.py`

**Overview:** No module docstring

**Key Imports:** `KMeans, numpy, pandas`

**Functions:**
- `load_and_expand()`: No description
- `detect_cycles()`: No description

---

## `analysis\decay_curve.py`

**Overview:** No module docstring

**Key Imports:** `KMeans, numpy, pandas`

**Functions:**
- `load_and_expand()`: No description
- `compute_lag_correlations()`: No description
- `compute_decay_curve()`: No description

---

## `analysis\diagnostic1.py`

**Overview:** No module docstring

**Key Imports:** `pandas`

*No functions defined (script may be procedural/linear execution).*

---

## `analysis\diagnostic2.py`

**Overview:** No module docstring

**Key Imports:** `pandas`

*No functions defined (script may be procedural/linear execution).*

---

## `analysis\drift_alignment.py`

**Overview:** No module docstring

**Key Imports:** `KMeans, numpy, pandas`

**Functions:**
- `load_hexagram_dataset()`: No description
- `load_alphabet()`: No description
- `build_stationary()`: No description
- `drift()`: No description

---

## `analysis\drift_timeline.py`

**Overview:** No module docstring

**Key Imports:** `KMeans, numpy, pandas`

**Functions:**
- `load_hp()`: No description
- `load_alpha()`: No description
- `build_stationary()`: No description
- `drift()`: No description

---

## `analysis\early_comparison.py`

**Overview:** No module docstring

**Key Imports:** `KMeans, numpy, pandas`

**Functions:**
- `load_and_expand()`: No description
- `build_chain()`: No description

---

## `analysis\epoch_comparison.py`

**Overview:** No module docstring

**Key Imports:** `matplotlib.pyplot, pandas`

*No functions defined (script may be procedural/linear execution).*

---

## `analysis\epoch_markov_analysis.py`

**Overview:** No module docstring

**Key Imports:** `numpy, pandas`

**Functions:**
- `build_markov_chain()`: Build transition matrix + stationary distribution.
- `top_attractors()`: No description
- `drift()`: No description

---

## `analysis\epoch_stats.py`

**Overview:** No module docstring

**Key Imports:** `chi2_contingency, numpy, pandas`

**Functions:**
- `permutation_test()`: No description

---

## `analysis\flatcluster_comparison.py`

**Overview:** No module docstring

**Key Imports:** `KMeans, numpy, pandas`

**Functions:**
- `load_hp()`: No description
- `load_alpha()`: No description
- `build_chain()`: No description

---

## `analysis\fractal_dimension.py`

**Overview:** No module docstring

**Key Imports:** `matplotlib.pyplot, numpy, pandas`

**Functions:**
- `load_points()`: No description
- `box_counting()`: Estimate fractal dimension using box-counting method.
- `plot_fractal_dimension()`: No description
- `main()`: No description

---

## `analysis\import_request.py`

**Overview:** No module docstring

**Key Imports:** `requests`

**Functions:**
- `fetch_quantum_byte()`: Fetches raw quantum entropy from ANU with updated headers and fallback endpoints.

---

## `analysis\invariance_test.py`

**Overview:** No module docstring

**Key Imports:** `KMeans, numpy, pandas`

**Functions:**
- `load_dataset()`: No description
- `build_stationary()`: No description
- `drift()`: No description

---

## `analysis\invertedcluster_comparison.py`

**Overview:** No module docstring

**Key Imports:** `KMeans, numpy, pandas`

**Functions:**
- `load_and_expand()`: No description
- `build_chain()`: No description

---

## `analysis\kmeans_k8.py`

**Overview:** No module docstring

**Key Imports:** `KMeans, matplotlib.pyplot, pandas`

*No functions defined (script may be procedural/linear execution).*

---

## `analysis\kmeans_probe.py`

**Overview:** No module docstring

**Key Imports:** `KMeans, matplotlib.pyplot, pandas`

*No functions defined (script may be procedural/linear execution).*

---

## `analysis\lag_correlation.py`

**Overview:** No module docstring

**Key Imports:** `KMeans, numpy, pandas`

**Functions:**
- `load_and_expand()`: No description
- `compute_lag_correlations()`: No description

---

## `analysis\line_randomization.py`

**Overview:** No module docstring

**Key Imports:** `KMeans, numpy, pandas`

**Functions:**
- `load_and_expand()`: No description
- `build_chain()`: No description

---

## `analysis\lyapunov_exponent.py`

**Overview:** No module docstring

**Key Imports:** `NearestNeighbors, matplotlib.pyplot, numpy, pandas`

**Functions:**
- `load_points()`: No description
- `estimate_lyapunov()`: Estimate the largest Lyapunov exponent using Rosenstein's algorithm.
- `plot_lyapunov()`: No description
- `main()`: No description

---

## `analysis\manifold_compare_all.py`

**Overview:** No module docstring

**Key Imports:** `csv, math, numpy`

**Functions:**
- `load_hex_dataset()`: No description
- `load_byte_dataset()`: No description
- `encode_states()`: No description
- `build_markov()`: No description
- `stationary_distribution()`: No description
- `curvature()`: No description
- `conditional_entropy()`: No description
- `drift()`: No description
- `kmeans()`: No description
- `cluster_weights()`: No description
- `cluster_curvature()`: No description
- `cluster_entropy()`: No description
- `top10()`: No description
- `basin_overlap()`: No description
- `analyze_manifold()`: No description
- `main()`: No description

---

## `analysis\manifold_metrics.py`

**Overview:** No module docstring

**Key Imports:** `KMeans, mean_squared_error, numpy, pandas`

**Functions:**
- `build_micro_states()`: No description
- `markov_matrix()`: No description
- `markov_feature_windows()`: No description
- `cluster_regimes()`: No description
- `regime_transition_matrix()`: No description
- `regime_stats()`: No description
- `simulate_regimes()`: No description
- `simulate_signal()`: No description
- `kl_divergence()`: No description
- `cross_entropy()`: No description
- `main()`: No description

---

## `analysis\manifold_model.py`

**Overview:** No module docstring

**Key Imports:** `KMeans, numpy, pandas`

**Functions:**
- `load_df()`: No description
- `build_micro_states()`: No description
- `markov_matrix()`: No description
- `markov_feature_windows()`: No description
- `cluster_regimes()`: No description
- `regime_transition_matrix()`: No description
- `regime_stats()`: No description
- `main()`: No description

---

## `analysis\manifold_simulation.py`

**Overview:** No module docstring

**Key Imports:** `mean_squared_error, numpy, pandas`

**Functions:**
- `simulate_regimes()`: No description
- `simulate_signal()`: No description

---

## `analysis\manifold_visualizer.py`

**Overview:** No module docstring

**Key Imports:** `matplotlib.pyplot, numpy, pandas, seaborn`

**Functions:**
- `simulate_regimes()`: No description
- `simulate_signal()`: No description

---

## `analysis\markov_chain.py`

**Overview:** No module docstring

**Key Imports:** `numpy, pandas`

*No functions defined (script may be procedural/linear execution).*

---

## `analysis\markovx64.py`

**Overview:** No module docstring

**Key Imports:** `KMeans, numpy, pandas`

**Functions:**
- `load_df()`: No description
- `build_states()`: No description
- `markov_matrix()`: No description
- `markov_curve()`: No description
- `cluster_features()`: No description
- `drift()`: No description
- `summarize_clusters()`: No description
- `main()`: No description

---

## `analysis\novelty_detector.py`

**Overview:** No module docstring

**Key Imports:** `numpy, pandas, time`

**Functions:**
- `safe_read()`: No description
- `compute_curve_novelty()`: No description
- `classify()`: No description
- `main()`: No description

---

## `analysis\null_model_collapse.py`

**Overview:** No module docstring

**Key Imports:** `KMeans, numpy, pandas`

**Functions:**
- `build_markov()`: No description
- `print_attractors()`: No description
- `drift()`: No description

---

## `analysis\oscillation.py`

**Overview:** No module docstring

**Key Imports:** `Counter, csv, os`

**Functions:**
- `load_hexagram_changing_counts()`: Load changing-line counts from hexagramdataset1.csv.
- `load_byte_bit_change_counts()`: Load bit-change counts from byte_manifold_dataset.csv.
- `print_distribution()`: No description
- `main()`: No description

---

## `analysis\permutation_test.py`

**Overview:** No module docstring

**Key Imports:** `chi2_contingency, numpy, pandas`

**Functions:**
- `permutation_test()`: No description

---

## `analysis\phase_space.py`

**Overview:** No module docstring

**Key Imports:** `KMeans, numpy, pandas`

**Functions:**
- `load_and_expand()`: No description
- `reconstruct_phase_space()`: No description

---

## `analysis\plot_umaptime.py`

**Overview:** No module docstring

**Key Imports:** `StandardScaler, matplotlib.pyplot, numpy, pandas, umap`

*No functions defined (script may be procedural/linear execution).*

---

## `analysis\probe_structuring.py`

**Overview:** No module docstring

**Key Imports:** `csv, sys`

**Functions:**
- `parse_hexagram_lines()`: No description
- `main()`: No description

---

## `analysis\quantum_stream.py`

**Overview:** No module docstring

**Key Imports:** `pandas, requests, time`

**Functions:**
- `fetch_quantum_vacuum_bytes()`: Fetches true quantum vacuum entropy from ANU Quantum Random Numbers API.

---

## `analysis\random_analysis.py`

**Overview:** No module docstring

**Key Imports:** `KMeans, numpy, pandas`

**Functions:**
- `load_structured_randomorg()`: No description
- `build_chain()`: No description
- `basin_strength()`: No description

---

## `analysis\raw_analyzer.py`

**Overview:** No module docstring

**Key Imports:** `csv, math, os`

**Functions:**
- `read_rows()`: No description
- `f()`: No description
- `analyze()`: No description
- `write_rows()`: No description
- `main()`: No description

---

## `analysis\raw_temporal_analysis.py`

**Overview:** No module docstring

**Key Imports:** `KMeans, StandardScaler, matplotlib.pyplot, numpy, os, pandas, sys`

**Functions:**
- `load_data()`: No description
- `build_markov_states()`: No description
- `compute_markov_transition_matrix()`: No description
- `build_markov_feature_curve()`: Slide over the state sequence and compute transition matrices
- `cluster_markov_features()`: No description
- `drift_analysis()`: Simple drift: split time into segments and compute mean/variance per segment.
- `plot_cluster_curve()`: No description
- `plot_drift()`: No description
- `main()`: No description

---

## `analysis\recurrence_analysis.py`

**Overview:** No module docstring

**Key Imports:** `KMeans, numpy, pandas`

**Functions:**
- `expand_hexagram_lines()`: No description

---

## `analysis\resolution_change.py`

**Overview:** No module docstring

**Key Imports:** `KMeans, numpy, pandas`

**Functions:**
- `load_and_expand()`: No description
- `build_chain()`: No description

---

## `analysis\run_diagnostic.py`

**Overview:** No module docstring

**Key Imports:** `pandas`

*No functions defined (script may be procedural/linear execution).*

---

## `analysis\run_hdbscan.py`

**Overview:** No module docstring

**Key Imports:** `StandardScaler, hdbscan, matplotlib.pyplot, numpy, pandas`

*No functions defined (script may be procedural/linear execution).*

---

## `analysis\run_umap.py`

**Overview:** No module docstring

**Key Imports:** `StandardScaler, matplotlib.pyplot, numpy, pandas, umap`

*No functions defined (script may be procedural/linear execution).*

---

## `analysis\shuffle_control.py`

**Overview:** No module docstring

**Key Imports:** `KMeans, numpy, pandas`

**Functions:**
- `expand_hexagram_lines()`: No description

---

## `analysis\space_density.py`

**Overview:** No module docstring

**Key Imports:** `Axes3D, matplotlib.pyplot, numpy, pandas`

**Functions:**
- `load_points()`: No description
- `compute_density()`: Compute 3D histogram density grid.
- `plot_density_2d()`: Plot 2D projections of the 3D density grid.
- `plot_density_3d()`: Render a 3D scatter of density peaks.
- `main()`: No description

---

## `analysis\stationary_distribution.py`

**Overview:** No module docstring

**Key Imports:** `KMeans, numpy, pandas`

**Functions:**
- `load_hexagrams()`: No description
- `build_chain()`: No description
- `stationary_delta()`: No description

---

## `analysis\synthetic_random.py`

**Overview:** No module docstring

**Key Imports:** `KMeans, numpy, pandas`

**Functions:**
- `load_and_expand()`: No description
- `build_chain()`: No description

---

## `analysis\temporal_curvature.py`

**Overview:** No module docstring

**Key Imports:** `KMeans, numpy, pandas`

**Functions:**
- `load_dataset()`: No description
- `build_stationary_and_curvature()`: No description
- `drift()`: No description

---

## `analysis\temporal_drift.py`

**Overview:** No module docstring

**Key Imports:** `matplotlib.pyplot, pandas`

*No functions defined (script may be procedural/linear execution).*

---

## `analysis\temporal_geometry_analysis.py`

**Overview:** No module docstring

**Key Imports:** `KMeans, datetime, math, matplotlib.pyplot, numpy, sys`

**Functions:**
- `parse_line()`: No description
- `load_entropy_stream()`: No description
- `compute_curvature()`: No description
- `detect_tunneling()`: No description
- `cluster_signatures()`: No description
- `drift_map()`: No description
- `plot_all()`: No description
- `main()`: No description

---

## `analysis\temporal_offset.py`

**Overview:** No module docstring

**Key Imports:** `json, os, subprocess`

**Functions:**
- `run_analysis()`: No description
- `parse_results()`: No description
- `main()`: No description

---

## `analysis\temporal_radar.py`

**Overview:** No module docstring

**Key Imports:** `matplotlib.pyplot, numpy, pandas, time`

**Functions:**
- `load_last_n()`: No description
- `main()`: No description

---

## `analysis\temporal_shuffle.py`

**Overview:** No module docstring

**Key Imports:** `KMeans, numpy, pandas`

**Functions:**
- `load_and_expand()`: No description
- `build_chain()`: No description

---

## `analysis\temporal_stability.py`

**Overview:** No module docstring

**Key Imports:** `KMeans, numpy, pandas`

**Functions:**
- `expand_hexagram_lines()`: No description
- `build_markov()`: No description
- `drift()`: No description

---

## `analysis\temporal_turbulance.py`

**Overview:** No module docstring

**Key Imports:** `KMeans, numpy, pandas`

**Functions:**
- `build_micro_states()`: No description
- `markov_matrix()`: No description
- `markov_feature_windows()`: No description
- `cluster_regimes()`: No description
- `simulate_regimes()`: No description
- `regime_transition_matrix()`: No description
- `regime_stats()`: No description
- `regime_entropy()`: No description
- `turbulence_metrics()`: No description
- `main()`: No description

---

## `analysis\temporal_weather.py`

**Overview:** No module docstring

**Key Imports:** `KMeans, numpy, pandas`

**Functions:**
- `build_micro_states()`: No description
- `markov_matrix()`: No description
- `markov_feature_windows()`: No description
- `cluster_regimes()`: No description
- `regime_entropy()`: No description
- `turbulence_metrics()`: No description
- `temporal_pressure()`: No description
- `detect_fronts()`: No description
- `main()`: No description

---

## `analysis\transition_entropy.py`

**Overview:** No module docstring

**Key Imports:** `KMeans, numpy, pandas`

**Functions:**
- `load_and_expand()`: No description
- `build_transition_matrix()`: No description
- `shannon_entropy()`: No description

---

## `analysis\tunneling_analyzer.py`

**Overview:** No module docstring

**Key Imports:** `Counter, csv, datetime, defaultdict`

**Functions:**
- `parse_hexagram_lines()`: No description
- `parse_changing_lines()`: No description
- `load_rows()`: No description
- `analyze_hexagrams()`: No description
- `analyze_system_state()`: No description
- `correlate_hexagrams_with_cpu()`: No description
- `main()`: No description
- `basic_stats()`: No description

---

## `analysis\tunneling_clusters.py`

**Overview:** No module docstring

**Key Imports:** `KMeans, numpy, pandas`

**Functions:**
- `load_and_expand()`: No description
- `build_clusters()`: No description

---

## `analysis\tunneling_compare.py`

**Overview:** No module docstring

**Key Imports:** `Counter, csv, defaultdict`

**Functions:**
- `parse_hexagram_lines()`: No description
- `parse_changing_lines()`: No description
- `load_rows()`: No description
- `analyze_dataset()`: No description
- `print_top_hexagrams()`: No description
- `print_distribution()`: No description
- `print_changing_distribution()`: No description
- `print_position_distribution()`: No description
- `compare_distributions()`: No description
- `main()`: No description

---

## `analysis\tunneling_entropy.py`

**Overview:** No module docstring

**Key Imports:** `matplotlib.pyplot, numpy, sounddevice`

**Functions:**
- `get_entropy_block()`: No description
- `compute_stats()`: No description
- `compute_fft()`: No description

---

## `analysis\tunneling_transition.py`

**Overview:** No module docstring

**Key Imports:** `csv, defaultdict, numpy`

**Functions:**
- `load_hexagrams_from_csv()`: No description
- `build_transition_counts()`: No description
- `normalize_counts()`: No description
- `top_transitions()`: No description
- `strongest_attractors()`: No description
- `forbidden_transitions()`: No description
- `basin_candidates()`: No description

---
