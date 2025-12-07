# train_gpt2_squad.py
# Fine-tune GPT-2 on the SQuAD dataset with a fixed answer format.

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
import torch

MODEL_NAME = "openai-community/gpt2"
OUTPUT_DIR = "./models/gpt2_squad_ft"

PREFIX = "That is a great question."
SUFFIX = "Let me know if you have any other questions."


def format_example(example):
    """Format a SQuAD sample into a single text block for causal LM training."""
    question = example["question"].strip()
    context = example["context"].strip()

    answers = example["answers"]["text"]
    answer_text = answers[0].strip() if len(answers) > 0 else "I am not sure."

    formatted_answer = f"{PREFIX} {answer_text} {SUFFIX}"

    text = (
        f"Question: {question}\n"
        f"Context: {context}\n"
        f"Answer: {formatted_answer}"
    )
    return {"text": text}


def main():
    # Load SQuAD
    squad = load_dataset("rajpurkar/squad")
    train_ds = squad["train"].shuffle(seed=42).select(range(5000))
    eval_ds = squad["validation"].shuffle(seed=42).select(range(500))

    train_ds = train_ds.map(format_example)
    eval_ds = eval_ds.map(format_example)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenization function
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
        )

    train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=train_ds.column_names)
    eval_tok = eval_ds.map(tokenize_fn, batched=True, remove_columns=eval_ds.column_names)

    # Use input_ids as labels for causal LM
    train_tok = train_tok.rename_column("input_ids", "labels")
    eval_tok = eval_tok.rename_column("input_ids", "labels")

    train_tok.set_format("torch")
    eval_tok.set_format("torch")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=200,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to=[],
    )

    # Simple data collator
    def collator(features):
        batch = {k: torch.stack([f[k] for f in features]) for k in features[0]}
        return batch

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
    )

    trainer.train()

    # Save fine-tuned model
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Saved fine-tuned model to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
