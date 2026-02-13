import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-14B-Instruct"

print(f"Loading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto", 
    torch_dtype=torch.float16,
    trust_remote_code=True
) 
print("Model Loaded. Generating...")
inputs = tokenizer("Hello?", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=10)
print(tokenizer.decode(outputs[0]))