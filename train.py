"""Fine-tune Helsinki-NLP/opus-mt-ja-en on manga-style dialogue (CPU).

Usage:
  python3 train.py [--pairs 20000] [--epochs 1] [--batch 16] [--steps N]
  --steps N caps training steps (quick benchmark: --steps 20).
Saves the fine-tuned model + tokenizer to model-manga-ja-en/.
"""
import argparse
import json
import os

import torch
from datasets import Dataset
from transformers import (MarianMTModel, MarianTokenizer,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          DataCollatorForSeq2Seq)

BASE = "Helsinki-NLP/opus-mt-ja-en"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "model-manga-ja-en")
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_pairs(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=20000)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=0, help="cap steps (0 = full)")
    ap.add_argument("--resume", action="store_true",
                    help="resume from latest checkpoint in train-out/")
    ap.add_argument("--lora", action="store_true",
                    help="train a small LoRA adapter instead of all weights "
                         "(~10x less memory; merged into a full model at save)")
    args = ap.parse_args()

    torch.set_num_threads(4)
    tok = MarianTokenizer.from_pretrained(BASE)
    model = MarianMTModel.from_pretrained(BASE)
    if args.lora:
        from peft import LoraConfig, get_peft_model, TaskType
        cfg = LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            task_type=TaskType.SEQ_2_SEQ_LM)
        model = get_peft_model(model, cfg)
        model.enable_input_require_grads()
        model.print_trainable_parameters()

    train_rows = load_pairs(f"{DATA}/train.jsonl")[:args.pairs]
    val_rows = load_pairs(f"{DATA}/val.jsonl")

    def tokenize(batch):
        src = tok(batch["ja"], truncation=True, max_length=64)
        tgt = tok(text_target=batch["en"], truncation=True, max_length=64)
        src["labels"] = tgt["input_ids"]
        return src

    train_ds = Dataset.from_list(train_rows).map(
        tokenize, batched=True, remove_columns=["ja", "en"])
    val_ds = Dataset.from_list(val_rows).map(
        tokenize, batched=True, remove_columns=["ja", "en"])

    targs = Seq2SeqTrainingArguments(
        output_dir=os.path.join(os.path.dirname(OUT), "train-out"),
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        num_train_epochs=args.epochs,
        max_steps=args.steps if args.steps > 0 else -1,
        logging_steps=25,
        eval_strategy="steps" if args.steps == 0 else "no",
        eval_steps=250,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        gradient_checkpointing=True,
        predict_with_generate=False,
        report_to=[],
        dataloader_num_workers=0,
        seed=42,
    )
    trainer = Seq2SeqTrainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tok, model=model),
    )
    ckpt = None
    if args.resume:
        out_dir = os.path.join(os.path.dirname(OUT), "train-out")
        if os.path.isdir(out_dir):
            ckpts = sorted(
                (d for d in os.listdir(out_dir)
                 if d.startswith("checkpoint-")),
                key=lambda d: int(d.split("-")[1]))
            if ckpts:
                ckpt = os.path.join(out_dir, ckpts[-1])
                print("RESUMING FROM:", ckpt)
    trainer.train(resume_from_checkpoint=ckpt)
    if args.lora:
        # Merge the adapter back into the base weights so the saved model
        # is a normal full model (no peft needed at inference time).
        model = model.merge_and_unload()
    trainer.save_model(OUT)
    tok.save_pretrained(OUT)
    print("SAVED:", OUT)


if __name__ == "__main__":
    main()
