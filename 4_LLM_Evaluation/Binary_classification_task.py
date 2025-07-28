

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import pandas as pd
import json
from tqdm import tqdm
import os
import numpy as np
from huggingface_hub import login
login(token="XXXX") 

model_id = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"

os.environ["TRANSFORMERS_CACHE"] = "XXXX"
os.environ["HF_HOME"] = "XXXX"


CHECKPOINT_FILE = "checkpoint_backup.json"
OUTPUT_FILE = "hal_levels_bin_output2_deepseekqwen.csv"
CHECKPOINT_FREQ = 10


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    return model, tokenizer

def save_checkpoint(index, predictions):
    with open(CHECKPOINT_FILE + ".tmp", "w") as f:
        json.dump({
            "last_index": index,
            "predictions": predictions
        }, f)
    os.replace(CHECKPOINT_FILE + ".tmp", CHECKPOINT_FILE)

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            data = json.load(f)
        return data["last_index"] + 1, data["predictions"]
    return 0, []


def zero_shot_classify(model, tokenizer, text):
    prompt = f"""You are given a research paper excerpt and determine if the paragraph contains hallucinations (factual inaccuracies/unsupported claims/nonsensical terms):

Title: {text['title']}
Abstract: {text['abstract']}
Context: {text['previous_paragraph']} [PARAGRAPH TO CHECK] {text['next_paragraph']}
Paragraph: "{text['best_modified_paragraph']}"

Does this paragraph contain hallucinations? Start the answer strictly with "YES" or "NO".
Answer:"""
    # truncation=True, max_length=2048
    inputs = tokenizer(prompt,return_tensors="pt").to(model.device)
    

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True
        )
    
   
    generated_sequence = outputs.sequences[0]
    full_output = tokenizer.decode(generated_sequence, skip_special_tokens=True)
    

    prompt_length = len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True))
    generated_text = full_output[prompt_length:].strip()
    generated_text = generated_text.upper()
    print(f"\nGenerated text: {repr(generated_text)}")  # Debug print
    
    
    valid_labels = ["YES", "NO"]
    if generated_text in valid_labels:
        print("Else: "+generated_text)
        return generated_text
    
    for label in valid_labels:
        if generated_text.startswith(label) or label.startswith(generated_text):
            print("Label: "+label)
            return label
       
    return "UNKNOWN"

def process_data():
    df = pd.read_csv("hal_levels_bin_output2_phi4.csv")
    model, tokenizer = load_model()
    
    start_idx, predictions = load_checkpoint()
    print(f"Resuming from row {start_idx}")

    try:
        for idx, row in tqdm(df.iloc[start_idx:].iterrows(), total=len(df)-start_idx):
            try:
                text = {
                    "title": row["title"],
                    "abstract": row["abstract"],
                    "previous_paragraph": row["previous_paragraph"],
                    "next_paragraph": row["next_paragraph"],
                    "best_modified_paragraph": row["best_modified_paragraph"]
                }
                
                pred = zero_shot_classify(model, tokenizer, text)
                predictions.append(pred)
                print(f"Granular: {row['label']}")
                print(f"Binary: {row['hal_nonhal_original']}")
                
                if (idx + 1) % CHECKPOINT_FREQ == 0 or idx == len(df) - 1:
                    save_checkpoint(idx, predictions)
                    
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    torch.cuda.empty_cache()
                    continue
                raise

    finally:
        df["deepseekqwen_binary"] = predictions + [None] * (len(df) - len(predictions))

        df.to_csv(OUTPUT_FILE, index=False)
        
        correct = (df["deepseekqwen_binary"] == df["hal_nonhal_original"]).sum()
        total = len(df)
        incorrect = total - correct

        print(f"Correct: {correct}, Incorrect: {incorrect}, Accuracy: {correct/total:.2%}")


if __name__ == "__main__":
    process_data()
