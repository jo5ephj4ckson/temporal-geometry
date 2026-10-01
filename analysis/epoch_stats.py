import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# === Load clustered dataset ===
df = pd.read_csv("clustered_output.csv")
df = df.sort_values("timestamp_unix")

# === Split into early and late epochs ===
split_point = df['timestamp_unix'].quantile(0.5)
early = df[df['timestamp_unix'] <= split_point]
late = df[df['timestamp_unix'] > split_point]

# === Count cluster frequencies ===
early_counts = early['cluster'].value_counts().sort_index()
late_counts = late['cluster'].value_counts().sort_index()

# === Build contingency table ===
contingency = pd.DataFrame({
    'Early': early_counts,
    'Late': late_counts
}).fillna(0)

print("Contingency Table:")
print(contingency)

# === Chi-square test ===
chi2, p, dof, expected = chi2_contingency(contingency)
print("\nChi-square Test Results:")
print(f"Chi2 statistic: {chi2}")
print(f"Degrees of freedom: {dof}")
print(f"P-value: {p}")

# === Cramér's V (effect size) ===
n = contingency.values.sum()
cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
print(f"Cramér's V (effect size): {cramers_v}")

# === Permutation test ===
def permutation_test(df, split_point, iterations=5000):
    observed = chi2
    count = 0
    for _ in range(iterations):
        shuffled = df.copy()
        shuffled['timestamp_unix'] = np.random.permutation(shuffled['timestamp_unix'])
        sp = shuffled['timestamp_unix'].quantile(0.5)
        e = shuffled[shuffled['timestamp_unix'] <= sp]['cluster'].value_counts().sort_index()
        l = shuffled[shuffled['timestamp_unix'] > sp]['cluster'].value_counts().sort_index()
        table = pd.DataFrame({'Early': e, 'Late': l}).fillna(0)
        chi2_perm, _, _, _ = chi2_contingency(table)
        if chi2_perm >= observed:
            count += 1
    return count / iterations

perm_p = permutation_test(df, split_point)
print(f"\nPermutation Test P-value: {perm_p}")
