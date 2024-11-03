# pyright: basic
from pydantic import BaseModel
from torch import device
from outlines import models, generate, samplers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "meta-llama/Llama-3.2-1B-Instruct"
# initialize model
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
llm = AutoModelForCausalLM.from_pretrained(
        model_id, 
        output_attentions=True, 
        quantization_config=quantization_config,
        device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = models.Transformers(llm, tokenizer)
sampler = samplers.greedy()

# classification
generator = generate.choice(model, ["skirt", "dress", "pen", "jacket"], sampler=sampler)
answer, logprobs = generator("Pick the odd word out: skirt, dress, pen, jacket", log_probs=True)

print(answer)
print(logprobs)
answer = generator("Pick the odd word out: skirt, dress, pen, jacket")
print(answer)

# regex
generator = generate.regex(
    model,
    r"((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)",
    sampler=sampler
)

prompt = "What is the IP address of the Google DNS servers? "
answer, logprobs = generator(prompt, max_tokens=30, log_probs=True)
print(answer)
print(logprobs)
answer = generator(prompt, max_tokens=30)
print(answer)

# json
class User(BaseModel):
    name: str
    last_name: str
    id: int

generator = generate.json(
    model, 
    User,
    sampler=sampler
)

result, logprobs = generator(
    "Create a user profile with the fields name, last_name and id. Please ensure it is valid json",
    log_probs=True
    )

print(result)
print(logprobs)
result = generator(
    "Create a user profile with the fields name, last_name and id"
)
print(result)

# type constraints
generator = generate.format(model, int)
answer, logprobs = generator(
    "When I was 6 my sister was half my age. Now I’m 70 how old is my sister?",
    log_probs=True)
print(answer)
print(logprobs)

generator = generate.format(model, int)
answer = generator("When I was 6 my sister was half my age. Now I’m 70 how old is my sister?")
print(answer)
