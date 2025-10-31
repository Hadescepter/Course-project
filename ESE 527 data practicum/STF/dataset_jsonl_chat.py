import os
import json
from torch.utils.data import Dataset

class JSONLChatDataset(Dataset):
    """
    用于加载 Chat 格式 JSONL 数据集
    每行应为 {"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}
    """

    def __init__(self, path, tokenizer, max_length=4096, eos_token=None):
        assert os.path.exists(path), f"Dataset not found: {path}"
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eos = eos_token or (tokenizer.eos_token if tokenizer.eos_token else "<|endoftext|>")
        self.samples = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                msgs = obj.get("messages")
                if not msgs:
                    prompt = obj.get("prompt")
                    if prompt:
                        text = str(prompt).rstrip() + self.eos
                        self.samples.append(text)
                    continue
                # 使用 Qwen 的模板自动拼接会话
                text = tokenizer.apply_chat_template(
                    msgs,
                    add_generation_prompt=False,
                    tokenize=False,
                )
                if not text.endswith(self.eos):
                    text = text + self.eos
                self.samples.append(text)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
        # 注意：这里不要 return_tensors="pt"
        # 让它返回 python 列表，方便 collator 统一 pad
        )
        return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        # 不在这里生成 labels，交给 collator 统一处理
        }