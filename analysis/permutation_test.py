import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# === Load clustered dataset ===
df = pd.read_csv("clustered_output.csv")
df = df.sort_values("timestamp_unix")

# === Split point for early/late ===
split_point = df['timestamp_unix'].quantile(0.5)

# === Compute observed chi-square ===
early = df[df['timestamp_unix'] <= split_point]
late = df[df['timestamp_unix'] > split_point]

early_counts = early['cluster'].value_counts().sort_index()
late_counts = late['cluster'].value_counts().sort_index()

contingency = pd.DataFrame({
    'Early': early_counts,
    'Late': late_counts
}).fillna(0)

observed_chi2, _, _, _ = chi2_contingency(contingency)

print(f"Observed Chi-square: {observed_chi2}")

# === Permutation test ===
def permutation_test(df, iterations=5000):
    count = 0
    for _ in range(iterations):
        shuffled = df.copy()
        shuffled['timestamp_unix'] = np.random.permutation(shuffled['timestamp_unix'])
        
        split = shuffled['timestamp_unix'].quantile(0.5)
        early_s = shuffled[shuffled['timestamp_unix'] <= split]['cluster'].value_counts().sort_index()
        late_s = shuffled[shuffled['timestamp_unix'] > split]['cluster'].value_counts().sort_index()
        
        table = pd.DataFrame({'Early': early_s, 'Late': late_s}).fillna(0)
        chi2_perm, _, _, _ = chi2_contingency(table)
        
        if chi2_perm >= observed_chi2:
            count += 1
    
    return count / iterations

perm_p = permutation_test(df, iterations=5000)
print(f"Permutation Test P-value: {perm_p}")
