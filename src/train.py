

import argparse
import json
import os
import time

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

from load_model import MODEL_NAME, load_model_and_tokenizer

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def to_conversational_dataset(records):
    
    examples = []
    for record in records:
        examples.append({
            "messages": [
                {"role": "user", "content": f"{record['instruction']}\n\n{record['input']}"},
                {"role": "assistant", "content": json.dumps(record["output"], ensure_ascii=False)},
            ]
        })
    return Dataset.from_list(examples)


class ProgressCallback(TrainerCallback):
    """Live tqdm progress bar + JSONL metrics log for training observability."""

    def __init__(self, log_path):
        self.log_path = log_path
        self.bar = None
        self.last_eval_loss = None
        self.last_lr = None
        self._log_file = open(log_path, "a", encoding="utf-8")

    def _vram_gb(self):
        if not torch.cuda.is_available():
            return 0.0, 0.0
        return torch.cuda.memory_allocated() / 1e9, torch.cuda.memory_reserved() / 1e9

    def _peak_vram_gb(self):
        if not torch.cuda.is_available():
            return 0.0, 0.0
        return torch.cuda.max_memory_allocated() / 1e9, torch.cuda.max_memory_reserved() / 1e9

    def _write_log(self, record):
        self._log_file.write(json.dumps(record) + "\n")
        self._log_file.flush()

    def _refresh_postfix(self, loss=None):
        if self.bar is None:
            return
        allocated, reserved = self._vram_gb()
        postfix = {}
        if loss is not None:
            postfix["loss"] = f"{loss:.4f}"
        if self.last_eval_loss is not None:
            postfix["val_loss"] = f"{self.last_eval_loss:.4f}"
        if self.last_lr is not None:
            postfix["lr"] = f"{self.last_lr:.2e}"
        postfix["vram_gb"] = f"{allocated:.1f}/{reserved:.1f}"
        self.bar.set_postfix(postfix)

    def on_train_begin(self, args, state, control, **kwargs):
        from tqdm.auto import tqdm
        self.bar = tqdm(total=state.max_steps, desc="Training", dynamic_ncols=True)

    def on_step_end(self, args, state, control, **kwargs):
        if self.bar is not None:
            self.bar.n = state.global_step
            self.bar.set_description(f"Epoch {state.epoch:.2f}")
            self.bar.refresh()

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        allocated, reserved = self._vram_gb()

        if "eval_loss" in logs:
            self.last_eval_loss = logs["eval_loss"]
        if "learning_rate" in logs:
            self.last_lr = logs["learning_rate"]

        peak_allocated, peak_reserved = self._peak_vram_gb()
        self._write_log({
            "timestamp": time.time(),
            "step": state.global_step,
            "epoch": state.epoch,
            "loss": logs.get("loss"),
            "eval_loss": logs.get("eval_loss"),
            "learning_rate": logs.get("learning_rate"),
            "vram_allocated_gb": allocated,
            "vram_reserved_gb": reserved,
            "vram_peak_allocated_gb": peak_allocated,
            "vram_peak_reserved_gb": peak_reserved,
        })

        self._refresh_postfix(loss=logs.get("loss"))

    def on_train_end(self, args, state, control, **kwargs):
        if self.bar is not None:
            self.bar.close()
        self._log_file.close()


def build_trainer(args, model, tokenizer, train_dataset, eval_dataset, log_path):
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, args.warmup_steps),
        optim="paged_adamw_8bit",
        bf16=True,
        max_length=args.max_seq_length,
        assistant_only_loss=True,
        logging_steps=args.logging_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        report_to=[],
        disable_tqdm=True,  # we drive our own bar via ProgressCallback
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        callbacks=[ProgressCallback(log_path)],
    )
    return trainer


def save_training_config(args, train_size, eval_size, path):
    config = {
        "model_name": MODEL_NAME,
        "train_data": args.train_data,
        "validation_data": args.validation_data,
        "train_examples": train_size,
        "validation_examples": eval_size,
        "max_train_samples": args.max_train_samples,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "max_seq_length": args.max_seq_length,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": LORA_TARGET_MODULES,
        "seed": args.seed,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-data", default="data/train.jsonl")
    parser.add_argument("--validation-data", default="data/validation.jsonl")
    parser.add_argument("--output-dir", default="models/it-support-lora")
    parser.add_argument("--max-train-samples", type=int, default=None,
                         help="Use only the first N training examples (for a quick smoke test).")
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_records = load_jsonl(args.train_data)
    if args.max_train_samples:
        train_records = train_records[:args.max_train_samples]
    train_dataset = to_conversational_dataset(train_records)

    eval_dataset = None
    if args.validation_data:
        eval_records = load_jsonl(args.validation_data)
        eval_dataset = to_conversational_dataset(eval_records)

    config = save_training_config(
        args, len(train_records), len(eval_dataset) if eval_dataset else 0,
        os.path.join(args.output_dir, "training_config.json"),
    )
    print("Training config:")
    print(json.dumps(config, indent=2))

    print(f"\nLoading {MODEL_NAME} in 4-bit...")
    model, tokenizer = load_model_and_tokenizer()

    log_path = os.path.join(args.output_dir, "training_log.jsonl")
    trainer = build_trainer(args, model, tokenizer, train_dataset, eval_dataset, log_path)

    print(f"\nStarting training: {len(train_records)} train / "
          f"{len(eval_dataset) if eval_dataset else 0} validation examples.")
    print(f"Metrics log: {log_path}\n")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nSaved LoRA adapter to {args.output_dir}")

    if torch.cuda.is_available():
        print(f"Peak VRAM this run: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB allocated / "
              f"{torch.cuda.max_memory_reserved() / 1e9:.2f} GB reserved")


if __name__ == "__main__":
    main()
