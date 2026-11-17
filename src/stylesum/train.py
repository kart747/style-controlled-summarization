"""Fine-tune FLAN-T5 on the multi-style corpus with control prefixes.

Every training example is ``"<style instruction>: <document>" -> <summary>``,
mixing all four styles in one model. At inference time changing the prefix is
the only thing needed to change the register of the output.

Usage::

    python -m stylesum.train --data data --out outputs/flan-t5-small-style
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

from .data import read_jsonl
from .model import DEFAULT_MODEL, MAX_SOURCE_TOKENS
from .styles import build_input

MAX_TARGET_TOKENS = 256


def tokenize_fn(tokenizer):
    def fn(batch):
        inputs = [build_input(d, s) for d, s in zip(batch["document"], batch["style"])]
        enc = tokenizer(inputs, max_length=MAX_SOURCE_TOKENS, truncation=True)
        enc["labels"] = tokenizer(
            text_target=batch["summary"], max_length=MAX_TARGET_TOKENS, truncation=True
        )["input_ids"]
        return enc

    return fn


def load_split(path: Path, tokenizer) -> Dataset:
    ds = Dataset.from_list(read_jsonl(path))
    return ds.map(tokenize_fn(tokenizer), batched=True, remove_columns=ds.column_names)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune a style-controlled summarizer")
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", type=Path, default=Path("outputs/flan-t5-small-style"))
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--optim", default="adafactor",
                    help="adafactor (T5's optimizer, tiny state - fits 4 GB GPUs) or adamw_torch")
    ap.add_argument("--grad-checkpointing", action="store_true",
                    help="trade compute for memory on small GPUs")
    a = ap.parse_args()

    set_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(a.model)
    train = load_split(a.data / "train.jsonl", tok)
    val = load_split(a.data / "val.jsonl", tok)
    print(f"train={len(train)} val={len(val)}")

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    args = Seq2SeqTrainingArguments(
        output_dir=str(a.out / "_tmp"),
        num_train_epochs=a.epochs,
        learning_rate=a.lr,
        lr_scheduler_type="linear",
        warmup_steps=0.05,  # float = ratio of total steps (transformers>=5)
        weight_decay=0.01,
        per_device_train_batch_size=a.batch,
        per_device_eval_batch_size=a.batch * 2,
        gradient_accumulation_steps=a.grad_accum,
        bf16=bf16,
        eval_strategy="epoch",
        save_strategy="no",  # disk is tight: only the final model is written
        logging_steps=50,
        report_to="none",
        train_sampling_strategy="group_by_length",
        optim=a.optim,
        gradient_checkpointing=a.grad_checkpointing,
        seed=a.seed,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=val,
        data_collator=DataCollatorForSeq2Seq(tok, model=model),
        processing_class=tok,
    )
    trainer.train()

    a.out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(a.out))
    tok.save_pretrained(str(a.out))
    history = [h for h in trainer.state.log_history if "loss" in h or "eval_loss" in h]
    (a.out / "train_log.json").write_text(json.dumps(history, indent=2))
    print("saved model ->", a.out)


if __name__ == "__main__":
    main()
