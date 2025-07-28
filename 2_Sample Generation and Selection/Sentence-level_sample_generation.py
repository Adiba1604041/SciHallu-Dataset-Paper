import pandas as pd 
df = pd.read_csv("Token_level_with_hallucinations.csv", low_memory=False)

# null value handling
df.fillna({'previous_paragraph': "There is no previous paragraph.", 'next_paragraph': "There is no next paragraph."}, inplace=True)

# parsing output
import re
def parse_hallucination_output(text, level):
    result = {}

    mod_paragraph_match = re.search(r'Modified_Paragraph:(.*?)(?:Perplexity:|$)', text, re.S)
    result['modified_paragraph'] = mod_paragraph_match.group(1).strip() if mod_paragraph_match else None

    perplexity_match = re.search(r'Perplexity:(.*?)(?:Explanation:|$)', text, re.S)
    result['perplexity'] = perplexity_match.group(1).strip() if perplexity_match else None

    if level=="token":
        explanation_match = re.search(r'Explanation:(.*?)(?:Number of substituted words from original paragraph:|$)', text, re.S)
        result['explanation'] = explanation_match.group(1).strip() if explanation_match else None

        num_sub_entities_match = re.search(r'Number of substituted words from original paragraph:(.*)', text, re.S)
        result['num_substituted_entities'] = num_sub_entities_match.group(1).strip() if num_sub_entities_match else None

    elif level=="sentence":
        explanation_match = re.search(r'Explanation:(.*?)(?:Number of substituted sentences from original paragraph:|$)', text, re.S)
        result['explanation'] = explanation_match.group(1).strip() if explanation_match else None

        num_sub_entities_match = re.search(r'Number of substituted sentences from original paragraph:(.*)', text, re.S)
        result['num_substituted_entities'] = num_sub_entities_match.group(1).strip() if num_sub_entities_match else None

    elif level=="paragraph":
        explanation_match = re.search(r'Explanation:(.*?)(?:Number of changes made to modify the original paragraph:|$)', text, re.S)
        result['explanation'] = explanation_match.group(1).strip() if explanation_match else None

        num_sub_entities_match = re.search(r'Number of changes made to modify the original paragraph:(.*)', text, re.S)
        result['num_substituted_entities'] = num_sub_entities_match.group(1).strip() if num_sub_entities_match else None

    return result


# semantic similarity
from sentence_transformers import SentenceTransformer, util 
model = SentenceTransformer('all-MiniLM-L6-v2')
def semantic_similarity(text1, text2):
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item()

import openai 
import time
import random

client = openai.OpenAI(api_key="XXXX")  

for idx, row in df.iterrows():
    label = row["label"]
    title = row["title"]
    abstract = row["abstract"]
    section = row["section"]
    paragraph = row["paragraph"] 
    previous_paragraph = row["previous_paragraph"]
    next_paragraph = row["next_paragraph"]
    domain =row["domain"] 
    random_number = random.randint(1, 3)

    system_prompt_sentence =f"""You are given a title, abstract, section, a current paragraph of the section and its adjacent two paragraphs (previous paragraph and next paragraph) from a research paper. 
        Your task is to introduce sentence-level hallucination to the current paragraph by replacing a few sentences with unrelated sentences (unrelated but not changing the theme of the overall paragraph), while keeping the overall structure of the paragraph intact. 
        The sentence-level hallucination can affect the logic or local context of the paragraph without changing the theme of the paragraph. The modified paragraph should not deviate from the global context. For global context refer to the title, abstract and adjacent paragraphs (if adjacent paragraphs exist).
        While replacing sentences, make sure they are not related to the title, abstract and section. Keep the sentences within the {domain} domain, but make them incorrect or misleading within that domain. This helps maintain realism while still introducing hallucination.
        You MUST NOT add any new sentences. The number of sentences in the modified paragraph must be EXACTLY the same as in the original paragraph. Every replaced sentence must correspond 1-to-1 with an original sentence. The substitution of sentences should introduce logical inconsistency or context inconsistency in the paragraph. The substitution should influence the paragraph's local coherence but should not disrupt the global coherence. 
        Introduce level {random_number} hallucination in the given paragraph. The hallucination level is defined below:
        
        1 — Low Noise (Minor Distortion):
        Substitutions are minimal and have little to no impact on the core semantics of the paragraph. The text remains coherent and truthful, with only slight inaccuracies or out-of-context sentences.
        → Barely noticeable, low risk of misunderstanding.

        2 — Medium Noise (Moderate Distortion):
        The paragraph contains noticeable but not overwhelming hallucinations. Important sentences may be replaced or altered, introducing partial misinformation or semantic drift.
        → Misleading without careful reading, moderate impact on factual integrity.

        3 — High Noise (Severe Distortion):
        Substantial hallucination significantly alters or corrupts the meaning of the paragraph. The substitutions make the text misleading, nonsensical, or factually incorrect.
        → High risk of misunderstanding, major semantic degradation.
         
         Provide the perplexity of the modified paragraph too. 
         Provide an explanation of why the modified paragraph has sentence-level hallucination and also mention all the sentences in the explanation that you modified from the original paragraph. Follow a common structure for explanation in all the responses. Maintain the following template in the output.
         provide the number of words that you replaced to introduce hallucination. Make sure you provide the right count.
         Modified_Paragraph:
         Perplexity:
         Explanation:
         Number of substituted sentences from original paragraph:"""
    

    if label=="Sentence-level hallucination":
        text_sentence_level = f"Title: {title}, Abstract: {abstract}, Setion: {section}, Current Paragraph: {paragraph}, Previous Paragraph: {previous_paragraph}, Next Paragraph: {next_paragraph}"
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt_sentence},

                    {"role": "user", "content": text_sentence_level}
                ],                                               
                temperature=0.7,
                max_tokens=8000,
                top_p=0.9,
                n=5
            )
            df.at[idx, f"noise_level"] = random_number
            for i in range(5):
                output_text = response.choices[i].message.content
                print(output_text)
                print("**************\n")

                parsed = parse_hallucination_output(output_text,"sentence")
                
                df.at[idx, f"modified_paragraph_{i+1}"] = parsed['modified_paragraph']
                df.at[idx, f"perplexity_{i+1}"] = parsed['perplexity']
                df.at[idx, f"semantic_similarity_{i+1}"] = semantic_similarity(paragraph, parsed['modified_paragraph'])
                df.at[idx, f"explanation_{i+1}"] = parsed['explanation']
                df.at[idx, f"num_substituted_entities_{i+1}"] = parsed['num_substituted_entities']
            time.sleep(1)

        except Exception as e:
            print(f"Error at row {idx}: {e}")
            df.at[idx, f"noise_level"] = "ERROR"
            for i in range(5):
                
                df.at[idx, f"modified_paragraph_{i+1}"] = "ERROR"
                df.at[idx, f"perplexity_{i+1}"] = "ERROR"
                df.at[idx, f"semantic_similarity_{i+1}"] = "ERROR"
                df.at[idx, f"explanation_{i+1}"] = "ERROR"
                df.at[idx, f"num_substituted_entities_{i+1}"] = "ERROR"
    if idx%10==0:
        df.to_csv("temporary_check.csv", index=False)
        
                
df.to_csv("Sentence_level_with_hallucinations.csv", index=False)
print("All responses generated and saved.")
