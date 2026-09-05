import torch
import torch.nn as nn
from torch.nn import functional as F
torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')
from tqdm import tqdm

# head_size: dimention of each head. calculated by n_embed//num_heads

class FeedForward(nn.Module):

    def __init__(self, n_embed, dropout_value):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout_value)
        )

    def forward(self, x):
        return self.net(x)

class AttentionHead(nn.Module):
    def __init__(self, n_embed, window_size, head_size, dropout_value):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(window_size, window_size)))

        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        B,T,C = x.shape

        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * C**-0.5 # attention scores
        wei = wei.masked_fill(self.tril[:T,:T] ==0, float('-inf')) # applying the masking
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v

        return out
    

class MultiHeadAttention(nn.Module):
    def __init__(self,n_embed, num_heads, window_size, head_size, dropout_value):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(n_embed, window_size, head_size, dropout_value) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class Block(nn.Module):

    def __init__(self, n_embed, num_head, window_size, head_size, dropout_value):
        super().__init__()
        self.sa = MultiHeadAttention(n_embed, num_head, window_size, head_size, dropout_value)
        self.ffwd = FeedForward(n_embed, dropout_value)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    # ffwd -> computation
    # att  -> comminucation

class TransformerModel(nn.Module):

    def __init__(self, n_embed, window_size, vocab_size, num_heads, num_transformer_blocks, dropout_value):
        super().__init__()
        if n_embed%num_heads == 0:
            head_size = n_embed//num_heads
        else:
            raise ValueError("n_embed is not devidable by num_heads. please choose different values.")
        self.window_size = window_size

        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(window_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, num_heads, window_size, head_size, dropout_value) for _ in range(num_transformer_blocks)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # idx and targets are both (B,T) tensors
        tok_emb = self.token_embedding_table(idx) # (B,T,C) 
        pos_emb = self.position_embedding_table(torch.arange(T))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x) #(B,T,vocab_size)
        
        if targets is None:
            loss = None

        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
        
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in tqdm(range(max_new_tokens)):
            # crop idx to the last window_size tokens
            idx_cond = idx[:, -self.window_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx