"""
train.py - GPT 模型训练流程
"""

import torch
import torch.nn as nn
import tiktoken
from dataset import create_dataloader, read_text_file
from gpt import GPTModel, generate_text_simple


# ==================== 辅助函数 ====================
def text_to_token_ids(text, tokenizer):
    """将文本转换为 token ids 张量，并增加 batch 维度"""
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    return torch.tensor(encoded).unsqueeze(0)  # [1, seq_len]


def token_ids_to_text(token_ids, tokenizer):
    """将 token ids 张量解码为文本（去除 batch 维度）"""
    return tokenizer.decode(token_ids.squeeze(0).tolist())


def calc_loss_batch(input_batch, target_batch, model, device):
    """计算单个 batch 的交叉熵损失"""
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)                     # [batch, seq_len, vocab]
    loss = nn.functional.cross_entropy(
        logits.flatten(0, 1),                       # [batch*seq_len, vocab]
        target_batch.flatten()                      # [batch*seq_len]
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """计算 DataLoader 上多个 batch 的平均损失（用于评估）"""
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    num_batches = num_batches if num_batches else len(data_loader)
    num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """评估模型在训练集和验证集上的损失"""
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    """给定起始文本，生成并打印后续内容"""
    model.eval()
    context_size = model.pos_emb.weight.shape[0]   # 模型的最大上下文长度
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))   # 将换行替换为空格便于显示
    model.train()


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    """主训练循环"""
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1

            # 定期评估
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # 每个 epoch 结束后生成一个样本
        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


# ==================== 主程序 ====================
if __name__ == "__main__":
    # 1. 模型配置（与 dataset 中的 max_length 保持一致）
    GPT_CONFIG_124M = {
        "vocab_size": 50257,           # GPT-2 的实际词汇表大小
        "context_length": 256,         # 必须与 DataLoader 的 max_length 一致
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }

    # 2. 准备数据
    tokenizer = tiktoken.get_encoding("gpt2")
    file_path = "the-verdict.txt"
    whole_text = read_text_file(file_path)

    train_ratio = 0.90
    split_idx = int(train_ratio * len(whole_text))
    train_text = whole_text[:split_idx]
    val_text = whole_text[split_idx:]

    train_loader = create_dataloader(
        text=train_text,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
    )

    val_loader = create_dataloader(
        text=val_text,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
    )

    # 3. 初始化模型、优化器、设备
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0004,
        weight_decay=0.1
    )

    # 4. 开始训练
    num_epochs = 10
    train_losses, val_losses, tokens_seen = train_model_simple(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        eval_freq=5,
        eval_iter=5,
        start_context="Every effort moves you",
        tokenizer=tokenizer
    )

    # 5.绘制损失曲线（需要 matplotlib）
    import matplotlib.pyplot as plt
    plt.plot(tokens_seen, train_losses, label="Train loss")
    plt.plot(tokens_seen, val_losses, label="Val loss")
    plt.legend()
    plt.show()