from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

LLM_MODEL_DIR = "./app/models/gpt2_squad_ft"

llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_DIR)
if llm_tokenizer.pad_token is None:
    llm_tokenizer.pad_token = llm_tokenizer.eos_token

llm_model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_DIR)

llm_device = "cuda" if torch.cuda.is_available() else "cpu"
llm_model.to(llm_device)
llm_model.eval()


class GenerateRequest(BaseModel):
    question: str
    max_new_tokens: int = 128


class GenerateResponse(BaseModel):
    question: str
    answer: str


@app.get("/")
def read_root():
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
def generate_answer(req: GenerateRequest):
    prompt = f"Question: {req.question}\nContext: \nAnswer:"

    inputs = llm_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(llm_device)

    with torch.no_grad():
        output_ids = llm_model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            pad_token_id=llm_tokenizer.eos_token_id,
        )

    full_text = llm_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    if "Answer:" in full_text:
        answer_text = full_text.split("Answer:", 1)[1].strip()
    else:
        answer_text = full_text.strip()

    return GenerateResponse(
        question=req.question,
        answer=answer_text,
    )
