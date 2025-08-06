from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

# Load fine-tuned Summarization model
model_path = "models/fine-tuned-Summarization-Bot"
print(f"Loading Summarization model from {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Set pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print("Summarization model loaded successfully!")

def summarize_text(text: str, max_new_tokens: int = 200): 
    prompt = f"Summarize the following text:\n\n{text}\n\nSummary:"
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
        return_attention_mask=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,          
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    try:
        summary = response.split("Summary:")[1].strip()
    except IndexError:
        summary = "Model failed to generate a valid summary."
    
    elapsed_time = time.time() - start_time

    return {"text": text, "summary": summary, "response_time": round(elapsed_time, 2)}