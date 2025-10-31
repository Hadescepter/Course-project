python train_qwen_lora.py \
  --model_name_or_path Qwen/Qwen2.5-1.5B \
  --data_path ./data/qwen_sft_test.jsonl \
  --output_dir ./qwen_lora_out \
  --num_train_epochs 2 \
  --learning_rate 2e-4 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --bf16
  