import pandas as pd
import numpy as np
t=pd.read_csv("normalized_perplexity.csv")
t["best_modified_paragraph"] = None
t["best_perplexity"] = None
t["best_semantic_similarity"] = None
t["best_abs_para_similarity"] = None
t["best_explanation"] = None
for idx, row in t.iterrows():
    min_weighted_sum = float('inf')
    max_weighted_sum = float('-inf')
    best_paragraph = None
    best_perplexity = None
    best_semantic_similarity= None
    best_abs_para_similarity= None
    best_explanation = None
    for i in range(1, 6):
        ws_col = f"weighted_sum_{i}"
        para_col = f"modified_paragraph_{i}"
        ppl_col = f"perplexity_{i}"
        ss_col=f"semantic_similarity_{i}"
        abs_ss_col=f"Abstract_Modified_Paragraph_Similarity_{i}"
        exp_col=f"explanation_{i}"

        try:
            ws_value = float(row[ws_col])
            if row["label"]=="No hallucination":
                if ws_value > max_weighted_sum:
                    max_weighted_sum = ws_value
                    best_paragraph = row[para_col]
                    best_perplexity = row[ppl_col]
                    best_semantic_similarity=row[ss_col]
                    best_abs_para_similarity=row[abs_ss_col]
                    best_explanation = row[exp_col]
                    
            else:
                if ws_value < min_weighted_sum:
                    min_weighted_sum = ws_value
                    best_paragraph = row[para_col]
                    best_perplexity = row[ppl_col]
                    best_semantic_similarity=row[ss_col]
                    best_abs_para_similarity=row[abs_ss_col]
                    best_explanation = row[exp_col]
                    
        except:
            continue 

    t.at[idx, "best_modified_paragraph"] = best_paragraph
    t.at[idx, "best_perplexity"] = best_perplexity
    t.at[idx, "best_semantic_similarity"] = best_semantic_similarity
    t.at[idx, "best_abs_para_similarity"] = best_abs_para_similarity
    t.at[idx, "best_explanation"] = best_explanation

t.to_csv("best_response.csv")
