"""
dataset.py - 用于语言模型的数据加载和嵌入模块
功能：
1. 从文本文件读取数据，进行 tokenization(使用 GPT-2 的 BPE 分词器)
2. 创建滑动窗口样本 (input, target) 用于 next-token 预测
3. 将样本打包成 mini-batch
4. 实现 token embedding + positional embedding
"""
import tiktoken          # OpenAI 的 BPE 分词器（用于 GPT-2）
import os
import urllib.request   # 下载文件
import re               # 正则表达式
import torch
from torch.utils.data import Dataset, DataLoader


class GPTDataset(Dataset):
    def __init__(self,text,tokenizer,max_length,stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(text)  # 将文本编码为 token ID 列表
        for i in range(0, len(token_ids) - max_length, stride):
            end = i + max_length
            input_chunck = token_ids[i:i+max_length]  # 输入序列
            target_chunck = token_ids[i+1:end+1] # 目标序列（输入序列的下一个 token）
            self.input_ids.append(torch.tensor(input_chunck))
            self.target_ids.append(torch.tensor(target_chunck))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader(text,batch_size,max_length,stride,shuffle=True,drop_last=True):
    """
    创建 DataLoader 对象，用于批量获取样本
    Args:
        text: str, 原始文本
        batch_size: int
        max_length: int, 每个样本的 token 数
        stride: int, 滑动步长
        shuffle: bool, 是否打乱顺序
        drop_last: bool, 若最后一批不足 batch_size 是否丢弃
    Returns:
        DataLoader 对象
    """
    # 初始化分词器（GPT-2 BPE）
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # 构建数据集
    dataset = GPTDataset(text, tokenizer, max_length, stride)
    
    # 创建 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0   # 单线程，便于调试
    )
    return dataloader

# ==========================嵌入层·==========================
def create_embedding_layer(input_tokens,vocab_size, embedding_dim):
    """
    对输入的 token 批次进行 token embedding + positional embedding
    Args:
        input_tokens: torch.Tensor, shape [batch_size, context_length], 每个元素是 token id
        vocab_size: int, 词汇表大小（用于 token embedding 层）
        embedding_dim: int, 嵌入向量的维度
    Returns:
        embeddings: torch.Tensor, shape [batch_size, context_length, embedding_dim]
    """
    batch_size, context_length = input_tokens.shape
    
    # Token Embedding
    token_embedding = torch.nn.Embedding(vocab_size, embedding_dim)#这里生成token_id和embedding的映射关系
    token_embedded = token_embedding(input_tokens)  # [batch_size, context_length, embedding_dim]
    
    # Positional Embedding
    # 创建可学习的位置嵌入表，形状 [context_length, embedding_dim]
    pos_embedding_layer = torch.nn.Embedding(context_length, embedding_dim)
    # 生成位置索引 0,1,2,...,context_length-1，形状 [context_length]
    positions = torch.arange(context_length)
    # 获取每个位置对应的嵌入向量，形状 [context_length, embedding_dim]
    pos_embeds = pos_embedding_layer(positions)
    
    # 广播相加：pos_embeds 会自动广播到 [batch_size, context_length, embedding_dim]
    return token_embedded + pos_embeds

# ==================== 3. 辅助函数 ====================
def download_file_if_not_exists(url, local_filename):
    """如果本地文件不存在，则从 URL 下载"""
    if not os.path.exists(local_filename):
        print(f"Downloading {local_filename} ...")
        urllib.request.urlretrieve(url, local_filename)
        print("Download finished.")
    else:
        print(f"File {local_filename} already exists.")
def read_text_file(filename):
    """读取文本文件并返回字符串"""
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()

# ==================== 4. 主程序 ====================
if __name__ == "__main__":
     # ---- 配置参数 ----
    DATA_URL = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
    DATA_FILE = "the-verdict.txt"
    BATCH_SIZE = 8
    CONTEXT_LENGTH = 4
    STRIDE = CONTEXT_LENGTH      # 窗口不重叠
    VOCAB_SIZE = 50257           # GPT-2 的词汇表大小
    EMBEDDING_DIM = 256

    # ---- 1. 获取文本数据 ----
    download_file_if_not_exists(DATA_URL, DATA_FILE)
    raw_text = read_text_file(DATA_FILE)
    print(f"Text length: {len(raw_text)} characters")
    
    # ---- 2. 创建 DataLoader ----
    dataloader = create_dataloader(
        text=raw_text,
        batch_size=BATCH_SIZE,
        max_length=CONTEXT_LENGTH,
        stride=STRIDE,
        shuffle=False
    )
    print(f"Number of batches: {len(dataloader)}")
    # ---- 3. 获取一个批次的输入和目标 ----
    data_iter = iter(dataloader)
    input_batch, target_batch = next(data_iter)
    print("Input batch shape:", input_batch.shape)   # [batch_size, context_length]
    print("Target batch shape:", target_batch.shape) # [batch_size, context_length]

    # ---- 4. 创建嵌入层并获取嵌入向量 ----
    torch.manual_seed(123)   # 固定随机种子，使结果可复现
    embeddings = create_embedding_layer(input_batch, VOCAB_SIZE, EMBEDDING_DIM)
    print("Embeddings shape:", embeddings.shape)  # [batch_size, context_length,