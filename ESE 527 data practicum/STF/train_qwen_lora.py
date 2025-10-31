import os
import argparse
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)
from peft import get_peft_model, set_peft_model_state_dict

from dataclasses import dataclass
from utils_trainable import print_trainable_parameters
from config_lora import build_lora_config
from dataset_jsonl_chat import JSONLChatDataset

@dataclass
class CausalLMCollator: 
    tokenizer: any
    def __call__(self, features):
        # 动态 pad 到 batch 内最长
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )
        batch["labels"] = batch["input_ids"].clone()
        # 把 pad 的位置屏蔽掉（不计入 loss）
        pad_mask = batch["attention_mask"] == 0
        batch["labels"][pad_mask] = -100
        return batch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./qwen_lora_out")
    parser.add_argument("--num_train_epochs", type=float, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    args = parser.parse_args()

    set_seed(42)


    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else None,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # LoRA
    lora_config = build_lora_config(
        r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout
    )
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)

    # dataset
    train_ds = JSONLChatDataset(
        args.data_path,
        tokenizer,
        eos_token=tokenizer.eos_token,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=["none"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        data_collator=CausalLMCollator(tokenizer),
    )

    # resume logic
    if args.resume_from_checkpoint:
        ckpt = os.path.join(args.resume_from_checkpoint, "adapter_model.bin")
        if os.path.exists(ckpt):
            adapters_weights = torch.load(ckpt, map_location="cpu")
            set_peft_model_state_dict(model, adapters_weights)
            print(f"Resumed from {ckpt}")

    # train
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(" Training complete!")


if __name__ == "__main__":
    main()