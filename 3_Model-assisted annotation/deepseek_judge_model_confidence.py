from transformers import AutoTokenizer, AutoModelForCausalLM
import torch # type: ignore

model_id = "deepseek-ai/deepseek-llm-7b-chat"
custom_cache = "/XXXX/models--deepseek-ai--deepseek-llm-7b-chat"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, cache_dir=custom_cache)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    cache_dir=custom_cache,
)

import pandas as pd # type: ignore
judge=pd.read_csv("Before_Annotation.csv")
judge=judge.head(100)


predictions = []
confidences = []

for idx, row in judge.iterrows():
    label = row["label"]
    title = row["title"]
    abstract = row["abstract"]
    paragraph = row["paragraph"] 
    modified_paragraph = row["best_modified_paragraph"]
    previous_paragraph = row["previous_paragraph"]
    next_paragraph = row["next_paragraph"]
    explanation = row["best_explanation"]


    token_message="the title, abstract, a paragraph, a modified version of that paragraph and an explanation"
    sen_para_message = "the title, abstract, a current paragraph, its adjacent two paragraphs (previous paragraph and next paragraph), a modified version of the current paragraph and an explanation"
    definition_token = "In token-level hallucination, some words in the paragraph are unrelated with respect to the title, abstract and paragraph itself."
    definition_sentence = "In sentence-level hallucination, some sentences in the paragraph is unrelated affecting the logic or local context of the paragraph without changing the theme of the paragraph. The hallucinated paragraph does not deviate from the global context where the global context refers to the title, abstract and adjacent paragraphs (if adjacent paragraphs exist)."
    definition_paragraph = "In paragraph-level hallucination, the whole paragraph deviates from the main **topic, intention, or argument** and no longer fits the global context. For global context refer to the title, abstract and adjacent paragraphs (if adjacent paragraphs exist)."

    if label == "Token-level hallucination":
        content_text= f"Title: {title}, Abstract: {abstract}, Original Paragraph: {paragraph}, Modified Paragraph: {modified_paragraph}, Explanation: {explanation}"
        passed_content = token_message
        definition = definition_token
    elif label == "Sentence-level hallucination":
        content_text= f"Title: {title}, Abstract: {abstract}, Original Paragraph: {paragraph}, Modified Paragraph: {modified_paragraph}, Previous Paragraph of the Original Paragraph: {previous_paragraph}, Next Paragraph of the Original Paragraph: {next_paragraph}, Explanation: {explanation}"
        passed_content = sen_para_message
        definition = definition_sentence
    elif label == "Paragraph-level hallucination":
        content_text= f"Title: {title}, Abstract: {abstract}, Original Paragraph: {paragraph}, Modified Paragraph: {modified_paragraph}, Previous Paragraph of the Original Paragraph: {previous_paragraph}, Next Paragraph of the Original Paragraph: {next_paragraph}, Explanation: {explanation}"
        passed_content = sen_para_message
        definition = definition_paragraph
    elif label == "No hallucination":
        content_text= f"Title: {title}, Abstract: {abstract}, Original Paragraph: {paragraph}, Modified Paragraph: {modified_paragraph}, Explanation: {explanation}"




    messages_nonhal = [
        {"role": "system", "content": """You are a judge who can analyze text from research papers with the help of good reasoning. 
        I will give you the title, abstract, original paragraph, a modified paragraph (non-hallucinated variant of the original paragraph)
        and an explanation where it is mentioned how the original paragraph has been modified preserving the original meaning. 
        You need to ONLY answer YES or No based on the following questions.
        Does the modified paragraph completely preserve the meaning of the original paragraph?
        Does the explanation clearly mention what modification are made in the original paragraph to make a non-hallucinated variant of the original paragraph?
        If both of the answers are YES for both of the questions, respond YES. Otherwise say NO. Do not add any explanation.
        """
        },
        {"role": "user", "content": content_text}
        ]


    messages_hal = [
    {"role": "system", "content": f"""You are a judge who can analyze text from research papers with the help of good reasoning. 
    I will give you {passed_content} where it is mentioned 
        how the original paragraph has been modified and why the modified paragraph introduces {label}. {definition}
        You need to ONLY answer YES or No based on the following questions.
        Does the modified paragraph introduce {label}?
        Does the explanation clearly mention what modification are made in the original paragraph to introduce hallucination?
        If both of the answers are YES for both of the questions, respond YES. Otherwise say NO. Don not add any explanation.
        """
        },
    {"role": "user", "content": content_text}
    ]

    if label=="No hallucination":
        message = messages_nonhal
    else:
        message = messages_hal
        
    prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

  
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  

    last_logits = logits[0, -1] 


    yes_token_id = tokenizer("YES", add_special_tokens=False)["input_ids"][0]
    no_token_id = tokenizer("NO", add_special_tokens=False)["input_ids"][0]



    probs = torch.softmax(last_logits[[yes_token_id, no_token_id]], dim=0)
    yes_prob, no_prob = probs.tolist()
   

    prediction = "YES" if yes_prob > no_prob else "NO"
    confidence = max(yes_prob, no_prob)
    predictions.append(prediction)
    confidences.append(confidence)

    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.4f}")

judge["model_prediction"] = predictions
judge["model_confidence"] = confidences
judge.to_csv("After_Model_Annotation.csv")
