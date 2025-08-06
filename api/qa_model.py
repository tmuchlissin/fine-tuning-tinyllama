from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

# Load fine-tuned QA model
model_path = "models/fine-tuned-QA-Bot"
print(f"Loading QA model from {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Set pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print("QA model loaded successfully!")

def get_answer(question: str, max_new_tokens: int = 200):  # 👈 Default 200, bukan 500
    prompt = f"Question: {question} Answer:"
    
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
            max_new_tokens=max_new_tokens,          # ✅ PASTIKAN PAKAI max_new_tokens
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response.replace(prompt, "").strip()
    elapsed_time = time.time() - start_time

    return {"question": question, "answer": answer, "response_time": round(elapsed_time, 2)}