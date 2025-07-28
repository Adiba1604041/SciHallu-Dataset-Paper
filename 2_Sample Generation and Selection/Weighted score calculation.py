import pandas as pd
import numpy as np
t=pd.read_csv("Final.csv")

# Collect all perplexities into a flat list
all_ppls = []
for i in range(1, 6):
    col = f"perplexity_{i}"

    all_ppls.extend(
        pd.to_numeric(t[col], errors='coerce').dropna().tolist()
    )


min_ppl = min(all_ppls)
max_ppl = max(all_ppls)
print(min_ppl, max_ppl)
# Function to apply conditional normalization
def conditional_normalize(x, label):
    if pd.isna(x):
        return None
    norm = (x - min_ppl) / (max_ppl - min_ppl)
    if label == "No hallucination":
        return 1 - norm
    return norm

# Apply conditional normalization and compute weighted sum
for i in range(1, 6):
    ppl_col = f"perplexity_{i}"
    sim_col = f"semantic_similarity_{i}"
    abs_sim_col = f"Abstract_Modified_Paragraph_Similarity_{i}"
    norm_col = f"converted_perplexity_{i}"
    weight_col = f"weighted_sum_{i}"

    # Convert perplexity to float
    t[ppl_col] = pd.to_numeric(t[ppl_col], errors='coerce')
    t[sim_col] = pd.to_numeric(t[sim_col], errors='coerce')
    t[abs_sim_col] = pd.to_numeric(t[abs_sim_col], errors='coerce')

    # Apply conditional normalization row-wise
    t[norm_col] = t.apply(lambda row: conditional_normalize(row[ppl_col], row['label']), axis=1)

    # Compute weighted sum
    t[weight_col] = (1/3) * t[norm_col] + (1/3) * t[sim_col] + (1/3) * t[abs_sim_col]


t.to_csv("normalized_perplexity.csv", index=False)
