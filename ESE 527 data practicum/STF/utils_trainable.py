
#打印模型中可训练参数数量和比例
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    pct = (100 * trainable_params / all_param) if all_param > 0 else 0
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {pct:.4f}"
    )