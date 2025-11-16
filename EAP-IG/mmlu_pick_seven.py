import pandas as pd

filepath = 'mmlu_performance.csv'
df = pd.read_csv(filepath)

# filter categories with performance >= 0.9
df_high = df[df['Score'] >= 0.9]

# drop unwanted categories
exclude = ["Astronomy - EM", "Marketing - EM", "Professional Medicine - EM"]
df_high = df_high[~df_high['Category'].isin(exclude)]

# randomly sample 7 rows (or fewer if not enough left)
sampled = df_high.sample(n=min(7, len(df_high)), random_state=2025)

print(sampled)
