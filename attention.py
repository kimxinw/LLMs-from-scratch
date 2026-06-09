"""
Attention is all you need! 
"""
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self,d_in,d_out,context_length,dropout,num_heads,qkv_bias=False):
        # d_in: int, 输入嵌入向量的维度
        # d_out: int, 输出嵌入向量的维度（通常等于 d_in）必须能被num_heads整除
        # context_length: int, 输入序列的长度（即上下文长度）
        # dropout: float, 注意力权重的 dropout 概率
        # num_heads: int, 注意力头的数量
        # qkv_bias: bool, 是否在查询、键、值的线性变换中使用偏置项
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.head_dim = d_out // num_heads
        self.num_heads = num_heads

        # 查询、键、值的投影层
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        
        # 输出线性层：将多头注意力的输出映射回原始维度
        self.out_proj = nn.Linear(d_out, d_in)
        self.dropout = nn.Dropout(dropout) 

        # Dropout 层
        self.attn_dropout = nn.Dropout(dropout)
        #因果掩码，防止模型在训练时看到未来的信息
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length),diagonal=1)) # 生成一个下三角矩阵，形状为 [context_length, context_length]

    def forward(self, x):
        """
        Args:
            x: [batch_size, num_tokens, d_in]
        Returns:
            context_vec: [batch_size, num_tokens, d_out]
        """
        b, num_tokens, d_in = x.shape
        #d_in 是输入嵌入向量的维度，应该与嵌入层的 embedding dimension 一致
        # 计算 Q, K, V
        queries = self.W_query(x)  # [b, num_tokens, d_out]
        keys    = self.W_key(x)
        values  = self.W_value(x)

        # 拆分为多头: [b, num_tokens, num_heads, head_dim] -> [b, num_heads, num_tokens, head_dim]
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys    = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values  = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        attn_scores = queries @ keys.transpose(2, 3)  # [b, num_heads, num_tokens, num_tokens]

        # 应用因果掩码（禁止看到未来 token）
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens],
            -torch.inf
        )# 将掩码位置的分数(上面self.mask把上三角设为1的掩码位置)设置为负无穷，这样在 softmax 归一化后，这些位置的权重将为零

        # Softmax 归一化（使用 head_dim 的平方根缩放）
        #当 head_dim 很大时,Q·K 算出来的点积值会变得很大,直接做 softmax 会让分布变得极端(几乎全部权重集中在一个位置),导致梯度非常小、训练不稳定。除以 √d_k 把数值拉回合理范围,让 softmax 更平滑、梯度更健康。
        attn_weights = torch.softmax(attn_scores / (self.head_dim ** 0.5), dim=-1)#dim=-1代表在最后一个维度上进行 softmax 归一化，即对每个查询对应的所有键的分数进行归一化，得到注意力权重
        attn_weights = self.dropout(attn_weights)

        # 加权求和
        context_vec = attn_weights @ values  # [b, num_heads, num_tokens, head_dim]

        # 合并多头: [b, num_tokens, num_heads, head_dim] -> [b, num_tokens, d_out]
        context_vec = context_vec.transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)

        # 最终线性变换
        return self.out_proj(context_vec)