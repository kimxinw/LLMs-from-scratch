"""
GPT 模型架构(Transformer Decoder)的实现
"""

import torch
import torch.nn as nn
from attention import MultiHeadAttention


class LayerNorm(nn.Module):
    """层归一化（可学习的缩放和偏移）"""
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        #eps是一个小常数，防止除以零的情况发生，确保数值稳定性
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        #scale 和 shift 是可学习的参数，分别用于缩放和偏移归一化后的输出，使模型能够恢复原始分布或学习更适合的分布
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        #均值和方差是沿着最后一个维度计算的，即对每个样本的嵌入向量进行归一化。keepdim=True 保持维度不变，以便后续的广播操作。均值0和方差1的标准化输出通过 scale 和 shift 进行线性变换，允许模型学习更适合的分布。
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """GELU 激活函数（比 ReLU 更平滑）"""
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * x ** 3)
        ))


class FeedForward(nn.Module):
    """前馈神经网络（扩展为 4 倍维度再投影回来）"""
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
            #两个线性层之间的 GELU 激活函数引入了非线性，使模型能够学习更复杂的函数映射
        )

    def forward(self, x):
        return self.layers(x)

#每个 Transformer 块包含 归一化、注意力机制、前馈网络(升维、激活、降维)和残差连接。Pre-LN 结构意味着在每个子层（注意力和前馈网络）之前进行层归一化，这有助于稳定训练过程并加速收敛。
class TransformerBlock(nn.Module):
    """单个 Transformer Decoder 块（Pre-LN 结构）"""
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.attn = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            dropout=cfg["drop_rate"],
            num_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.dropout = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # 第一个残差：注意力 + 残差连接
        shortcut = x
        x = self.norm1(x)#前馈网络和注意力机制之前的层归一化有助于稳定训练过程，防止梯度消失或爆炸，并加速模型的收敛。
        x = self.attn(x)
        x = shortcut + self.dropout(x) #dropout 在残差连接后应用，有助于正则化模型，减少过拟合的风险。通过在训练过程中随机丢弃一些连接，dropout 强制模型学习更鲁棒的特征表示。

        # 第二个残差：前馈网络 + 残差连接
        shortcut = x
        x = self.norm2(x)#
        x = self.ff(x)
        x = shortcut + self.dropout(x)
        return x


class GPTModel(nn.Module):
    """完整的 GPT 模型"""
    def __init__(self, cfg):
        super().__init__()
        self.token_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])

        # 堆叠多个 Transformer 块
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, x):
        """
        Args:
            x: [batch_size, context_length] (token ids)
        Returns:
            logits: [batch_size, context_length, vocab_size]
        """
        batch_size, context_len = x.shape

        # Token embedding + Positional embedding
        token_embeds = self.token_emb(x)  # [b, context_len, emb_dim]
        pos_embeds = self.pos_emb(torch.arange(context_len, device=x.device))  # [context_len, emb_dim]
        x = token_embeds + pos_embeds
        x = self.dropout(x)

        # 通过 Transformer 块
        x = self.trf_blocks(x)        # [b, context_len, emb_dim]
        x = self.final_norm(x)

        # 输出投影到词汇表大小
        logits = self.out_head(x)     # [b, context_len, vocab_size]
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    贪心解码生成文本
    Args:
        model: GPTModel
        idx: [batch_size, current_seq_len] 起始 token ids
        max_new_tokens: 要生成的 token 数量
        context_size: 模型支持的最大上下文长度
    Returns:
        idx: [batch_size, current_seq_len + max_new_tokens]
    """
    for _ in range(max_new_tokens):
        # 如果序列超过 context_size，则截断到最近的 context_size 个 token
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)          # [batch_size, context_len, vocab_size]

        # 只取最后一个时间步的输出
        logits_last = logits[:, -1, :]        # [batch_size, vocab_size]
        probs = torch.softmax(logits_last, dim=-1)
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)  # [batch_size, 1]

        # 拼接到序列末尾
        idx = torch.cat((idx, idx_next), dim=1)#[batch_size, current_seq_len+1]

    return idx

