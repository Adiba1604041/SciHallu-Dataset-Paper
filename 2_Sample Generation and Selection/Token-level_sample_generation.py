import pandas as pd 
df = pd.read_csv("combined_dataset_v2_with_labels.csv", low_memory=False)

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
import tiktoken
client = openai.OpenAI(api_key="XXXX")  
#enc = tiktoken.encoding_for_model("gpt-4o-mini")

for idx, row in df.iterrows():
    label = row["label"]
    title = row["title"]
    abstract = row["abstract"]
    section = row["section"]
    paragraph = row["paragraph"] 
    domain =row["domain"] 
    random_number = random.randint(1, 3)

    system_prompt_token =f"""You are given a title, abstract, section and a paragraph of the section from a research paper. Your task is to introduce token-level hallucination to the paragraph by replacing a few individual words with unrelated words (unrelated but not absurd words), while keeping the overall structure and most of the sentences intact. 
        While replacing words, make sure they are not related to the title, abstract and section. Keep the words within the {domain} domain, but make them incorrect, misleading, or out-of-context within that domain. This helps maintain realism while still introducing hallucination.
        Do not add or remove entire phrases or sentences—just substitute some existing words with incorrect or out-of-context ones. The substitution of word may include fake named entity, wrong number, fake acronym and many more. Do not substitute stop words. Introduce level {random_number} hallucination in the given paragraph. The hallucination level is defined below:
        
        1 — Low Noise (Minor Distortion):
        Substitutions are minimal and have little to no impact on the core semantics of the paragraph. The text remains coherent and truthful, with only slight inaccuracies or out-of-place terms.
        → Barely noticeable, low risk of misunderstanding.

        2 — Medium Noise (Moderate Distortion):
        The paragraph contains noticeable but not overwhelming hallucinations. Key terms may be replaced or altered, introducing partial misinformation or semantic drift.
        → Misleading without careful reading, moderate impact on factual integrity.

        3 — High Noise (Severe Distortion):
        Substantial hallucination significantly alters or corrupts the meaning of the paragraph. The substitutions make the text misleading, nonsensical, or factually incorrect.
        → High risk of misunderstanding, major semantic degradation.
         
         Provide the perplexity of the modified paragraph too. 
         Provide an explanation of why the modified paragraph has token-level hallucination and also mention all the words in the explanation that you modified from the original paragraph. Follow a common structure for explanation in all the responses. Maintain the following template in the output.
         provide the number of words that you replaced to introduce hallucination. Make sure you provide the right count.
         Modified_Paragraph:
         Perplexity:
         Explanation:
         Number of substituted words from original paragraph:"""
    
    if label=="Token-level hallucination":
        text_token_level = f"Title: {title}, Abstract: {abstract}, Setion: {section}, Paragraph: {paragraph}"
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt_token},

                    {"role": "user", "content": text_token_level}
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
                #tokens = enc.encode(output_text)
                #print(f"Choice {i+1} token count: {len(tokens)}\n")
                parsed = parse_hallucination_output(output_text,"token")
                
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
                
df.to_csv("Token_level_with_hallucinations.csv", index=False)
print("All responses generated and saved.")
