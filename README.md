Overview

This dataset is constructed to support fine-grained hallucination detection in scientific writing. It is split into two major parts:

---
1. Part 1 (Common Information)

This section contains information that is common across multiple instances.

Columns:
- unique ID for mapping
- domain: The academic domain or field (e.g., Computer Science, Health Science)
- venue: The name of the journal or conference
- title: Title of the research paper
- abstract: The abstract summarizes the paper
- section: The section name from where the paragraph was extracted(e.g., Introduction, Results)

These values are mapped to multiple rows in Part 2 of the dataset to avoid redundancy.

---

2. Part 2  (Merge Part-2a,2b,2c) (Instance-Level Data)

Part 2 has been divided into 3 chunks to maintain the file size restriction in GitHub. Merge them to get the entire Part 2. This part contains individual rows corresponding to the detailed part of each paragraph instance.

Columns:
- unique ID for mapping
- paragraph: The paragraph under study
- previous_paragraph, next_paragraph: Adjacent context paragraphs
- label: Annotation label (e.g., Token-level hallucination, Sentence-level hallucination, Paragraph-level hallucination, No hallucination)
- noise_level: Indicates the intensity of alteration
- best_modified_paragraph: The best model-generated variant of the paragraph
- best_explanation: Explanation generated for the modification
- best_perplexity (before normalization)
- best_semantic_similarity (semantic similarity between original and modified paragraph)
- best_abs_para_similarity (semantic similarity between abstract and original paragraph)

---

Mapping Between Part 1 and Part 2

Each row in Part 2 of the dataset is linked to metadata via unique ID:
- title
- abstract
- section
- domain
- venue

Joining Part 1 and Part 2 dataframes on the unique ID results in the complete dataset.

---

