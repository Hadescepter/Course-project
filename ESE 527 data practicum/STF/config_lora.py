from peft import LoraConfig, TaskType

# Qwen / LLaMA 系列的常用 LoRA 模块映射
TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING = {
    "qwen": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
}

def build_lora_config(r=8, alpha=32, dropout=0.1, bias="none"):
    """
    生成 LoRA 配置
    """
    target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING["qwen"]
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias=bias,
    )