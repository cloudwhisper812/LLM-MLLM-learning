## Transformer Block

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 这里的w其实是多个heads的，矩阵计算方便和在一起了
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        # (batch, seq, d_model) -> (batch, seq, d_model)
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # 拆分成多个 Head
        # (batch, seq, num_heads, d_k)
        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)

        context = torch.matmul(attn_weights, v)
        # 拼接所有 Head (Concat)
        # (batch, seq, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # [Step 5] 经过 W_o 这个全连接层进行“大揉搓”
        # 没有这一层，Head 之间就是孤立的
        output = self.w_o(context)

        return output

class SwiGLUMLP(nn.Module):
    def __init__(self, d_model, intermediate_size):
        super().__init__()
        self.act = nn.SiLU()
        self.w_gate = nn.Linear(d_model, intermediate_size, bias=False)
        self.w_up = nn.Linear(d_model, intermediate_size, bias=False)
        self.w_down = nn.Linear(intermediate_size, d_model, biase=False)
    def forward(self, x):
        # 这里和我一开始理解的顺序不一样，需要看一下，我理解的 self.w_gate(x) * self.act(self.w_up(x)) 
        x = self.w_up(x) * self.act(self.w_gate(x)) 
        return self.w_down(x)


class TransformeBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.ln_1 = nn.RMSNorm(d_model)
        self.attn = MuliHeadAttention(d_model, num_heads)
        self.ln2 = nn.RMSNorm(d_model)
        sle.mlp = SwiGLUMLP(d_model, intermediate_size = d_model * 4)
    def forward(self, x):
        # pre-norm 优于post norm
        x = x + self.attn(self.ln_1(x))

        # 这里的mlp有两层，中间的激活函数是transformer block中唯一一次
        x = x + self.mlp(self.ln2(x))
```
