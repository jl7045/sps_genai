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

# 预训练模型名称
MODEL_NAME = "openai-community/gpt2"

# 微调后模型的保存位置（相对项目根目录）
OUTPUT_DIR = "./app/models/gpt2_squad_ft"

# 回答格式要求
PREFIX = "That is a great question."
SUFFIX = "Let me know if you have any other questions."


def format_example(example):
    """把一条 SQuAD 样本转成训练用的文本。"""
    question = example["question"].strip()
    context = example["context"].strip()

    answers = example["answers"]["text"]
    answer_text = answers[0].strip() if len(answers) > 0 else "I am not sure."

    # 按作业要求拼接前后缀
    formatted_answer = f"{PREFIX} {answer_text} {SUFFIX}"

    text = (
        f"Question: {question}\n"
        f"Context: {context}\n"
        f"Answer: {formatted_answer}"
    )
    return {"text": text}


def main():
    # 1. 加载 SQuAD 数据集
    print(">>> Loading SQuAD dataset...")
    squad = load_dataset("rajpurkar/squad")
    # 先抽样一部分，避免训练太慢/卡死，可按需调大
    train_ds = squad["train"].shuffle(seed=42).select(range(5000))
    eval_ds = squad["validation"].shuffle(seed=42).select(range(500))

    train_ds = train_ds.map(format_example)
    eval_ds = eval_ds.map(format_example)

    # 2. 加载 tokenizer
    print(">>> Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. tokenization
    def tokenize_fn(batch):
        enc = tokenizer(
            batch["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
        )
        # 把 input_ids 复制一份作为 labels（因果语言模型训练）
        enc["labels"] = enc["input_ids"].copy()
        return enc

    print(">>> Tokenizing train dataset...")
    train_tok = train_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=train_ds.column_names,
    )

    print(">>> Tokenizing eval dataset...")
    eval_tok = eval_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=eval_ds.column_names,
    )

    train_tok.set_format("torch")
    eval_tok.set_format("torch")

    # 4. 加载模型
    print(">>> Loading model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 5. 训练参数
    print(">>> Setting up training arguments...")
    training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    logging_steps=50,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=200,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    report_to=[],              # 不上报到 wandb 等
    dataloader_num_workers=0,  # Windows 下避免 DataLoader 多进程死锁
)


    # 6. 简单的 data collator
    def collator(features):
        batch = {k: torch.stack([f[k] for f in features]) for k in features[0]}
        return batch

    # 7. Trainer
    print(">>> Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
    )

    # 8. 开始训练
    print(">>> Start training...")
    trainer.train()

    # 9. 保存模型和 tokenizer
    print(f">>> Saving fine-tuned model to: {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(">>> Done.")


if __name__ == "__main__":
    main()
