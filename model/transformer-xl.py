import torch
import torch.nn as nn
import torch.nn.functional as F

class StreamingMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.scaling = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_proj.weight)
        self.out_proj.bias.data.fill_(0)

    def _shape(self, tensor, seq_len, bsz):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False, attn_mask=None):
        bsz, tgt_len, embed_dim = query.size()
        src_len = key.size(1)
        
        # Project and reshape the queries, keys, and values
        q, k, v = self.qkv_proj(query).chunk(3, dim=-1)
        
        q = self._shape(q, tgt_len, bsz) * self.scaling
        k = self._shape(k, src_len, bsz)
        v = self._shape(v, src_len, bsz)

        # Compute the attention scores
        attn_output_weights = torch.matmul(q, k.transpose(-2, -1))
        if attn_mask is not None:
            attn_output_weights += attn_mask
        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            attn_output_weights = attn_output_weights.view(bsz * self.num_heads, tgt_len, src_len)
        
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = self.dropout(attn_output_weights)
        
        # Compute the attention output
        attn_output = torch.matmul(attn_output_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, embed_dim)
        attn_output = self.out_proj(attn_output)
        
        if need_weights:
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len).sum(dim=1) / self.num_heads
            return attn_output, attn_output_weights
        else:
            return attn_output, ()

class MusicFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(MusicFeedForward, self).__init__()
        # 扩大模型容量，d_ff 通常是 d_model 的 4 倍
        self.linear1 = nn.Linear(d_model, d_ff)
        # 引入更多的非线性激活函数，以处理复杂的音乐模式
        self.relu1 = nn.ReLU()
        # 可以考虑添加更多的隐藏层
        self.linear2 = nn.Linear(d_ff, d_ff)
        self.relu2 = nn.ReLU()
        # 将数据映射回原来的维度
        self.linear3 = nn.Linear(d_ff, d_model)
        # 额外的层归一化和丢弃层，以稳定训练过程和防止过拟合
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.dropout(self.relu1(self.linear1(x)))
        x = self.dropout(self.relu2(self.linear2(x)))
        x = self.linear3(x)
        # 应用层归一化和残差连接
        x = self.layer_norm(x + residual)
        return x

class TransformerXLBlock(nn.Module):
    def __init__(self, embed_size, heads, depth, seq_length, mem_length):
        super(TransformerXLBlock, self).__init__()
        self.attention = StreamingMultiheadAttention(embed_size, heads)
        self.feed_forward = MusicFeedForward(embed_size)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.seq_length = seq_length
        self.mem_length = mem_length

    def forward(self, x, mem):
        # Combine input with memory
        cat = torch.cat([mem, x], dim=1)
        attn_out, _ = self.attention(cat, cat, cat)
        x = self.norm1(attn_out + x)
        ff_out = self.feed_forward(x)
        x = self.norm2(ff_out + x)
        # Update memory
        new_mem = cat[:, -self.mem_length:]
        return x, new_mem

class TransformerXL(nn.Module):
    def __init__(self, embed_size, heads, depth, seq_length, mem_length, vocab_size):
        super(TransformerXL, self).__init__()
        self.layers = nn.ModuleList([TransformerXLBlock(embed_size, heads, depth, seq_length, mem_length) for _ in range(depth)])
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_length + mem_length, embed_size))
        self.to_vocab = nn.Linear(embed_size, vocab_size)
        self.seq_length = seq_length
        self.mem_length = mem_length

    def forward(self, x, mems=None):
        if mems is None:
            mems = [torch.zeros((x.size(0), self.mem_length, self.embed_size)).to(x.device) for _ in range(len(self.layers))]
        
        x = self.embed(x)
        x += self.pos_embed[:, -self.seq_length:]

        new_mems = []
        for i, layer in enumerate(self.layers):
            x, new_mem = layer(x, mems[i])
            new_mems.append(new_mem)

        logits = self.to_vocab(x)
        return logits, new_mems

# Example usage:
# vocab_size = 128 (for example, MIDI note range)
model = TransformerXL(embed_size=512, heads=8, depth=6, seq_length=512, mem_length=512, vocab_size=128)

# Dummy inputs
x = torch.randint(0, 128, (32, 512))  # (batch_size, seq_length)
mems = None  # No memory at the beginning

# Forward pass
output, new_mems = model(x, mems)