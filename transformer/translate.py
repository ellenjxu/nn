# My implementation of Encoder and decoder for translation
# https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/train.de.gz Multi30k

# Input: German text
# Output: English text

import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken as tk
import random

# hyperparameters
batch_size = 8
block_size = 8 # 64
max_iters = 500 # 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 1 
n_embd = 64 
n_head = 4 
n_layer = 4
dropout = 0.2
# ------------

torch.manual_seed(1203892)

with open('train.de', 'r', encoding='utf-8') as f:
    de_text = f.read()
with open('train.en', 'r', encoding='utf-8') as f:
    en_text = f.read()
    
# get tokens
enc = tk.get_encoding('gpt2')
vocab_size = enc.max_token_value + 1

# <START> and <END> tokens
de = [torch.tensor(enc.encode('<START> ' + d.replace('\n', '<END>'))) for d in de_text]
en = [torch.tensor(enc.encode('<START> ' + e.replace('\n', '<END>'))) for e in en_text]

# add padding to make all de sequences the same length
max_len = max([len(d) for d in de])
de = [F.pad(d, (0, max_len - len(d))) for d in de]
en = [F.pad(e, (0, block_size - len(e))) for e in en] # pad en to block size

# train-val split
z = list(zip(de, en))
random.shuffle(z)
n = int(0.9*len(z))
train_data = z[:n]
val_data = z[n:]

def get_batch(split):
  # generate a small batch of data of inputs x and targets y
  
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data), (batch_size,)) # randomly select batch_size sentences
  de = torch.stack([data[i][0] for i in ix])
  en = [data[i][1] for i in ix]
  en_labels = [data[i][1][1:] for i in ix]
  
  # randomly select block_size tokens from each sentence
  en_b = []
  en_labels_b = []
  for s1, s2 in zip(en, en_labels):
    if len(s1) == block_size:
      en_b.append(s1)
      en_labels_b.append(s2)
      continue
    i = torch.randint(len(s1)-block_size, (1,))[0]
    en_b.append(s1[i:i+block_size])
    en_labels_b.append(s2[i:i+block_size])
  
  en_b = torch.stack(en_b)
  en_labels_b = torch.stack(en_labels_b)
  
  return (de.to(device), en_b.to(device)), en_labels_b.to(device)

class MultiHeadAttention(nn.Module):
  """ one head of self-attention """

  def __init__(self, num_heads):
      super().__init__()
      self.key = nn.Linear(n_embd, n_embd, bias=False)
      self.query = nn.Linear(n_embd, n_embd, bias=False)
      self.value = nn.Linear(n_embd, n_embd, bias=False)
      self.proj = nn.Linear(n_embd, n_embd)
      self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
      self.dropout = nn.Dropout(dropout)
      self.num_heads = num_heads

  def forward(self, q, k, v, mask=None):
      k = self.split(self.key(k)) # (B, T, C) -> (B, H, T, C/H)
      q = self.split(self.query(q))
      # compute attention scores ("affinities") and normalize by head size
      wei = q @ k.transpose(-2,-1) * (n_embd // self.num_heads) ** -0.5 # (B, H, T, C/H) @ (B, H, C/H, T) -> (B, H, T, T
      if mask:
        wei = wei.masked_fill(self.tril == 0, float('-inf')) # (B, H, T, T)
      wei = F.softmax(wei, dim=-1) # (B, H, T, T)
      wei = self.dropout(wei)
      # perform the weighted aggregation of the values
      v = self.split(self.value(v))
      # use q below because q will always have the same shape as x
      out = (wei @ v).transpose(1,2).contiguous().view(q.shape[0], q.shape[2], -1) # (B, H, T, T) @ (B, H, T, C/H) -> (B, T, H, C/H) -> (B, T, C)
      return out

  def split(self, x): # split the last dimension into num_heads
      B,T,C = x.shape
      return x.view(B, T, self.num_heads, C // self.num_heads).transpose(1,2)

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    
class EncoderBlock(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        # head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = self.ln1(x)
        x = x + self.sa(q=x, k=x, v=x, mask=None) # residual skip connections +
        x = self.ln2(x)
        x = x + self.ffwd(x)
        return x
      
class DecoderBlock(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        # head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head)
        self.eda = MultiHeadAttention(n_head)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)

    def forward(self, x, enc):
        x = self.ln1(x)
        x = x + self.sa(q=x, k=x, v=x, mask=True)
        # encoder-decoder attention
        x = self.ln2(x)
        x = x + self.eda(q=x, k=enc, v=enc, mask=None)
        x = self.ln3(x)
        x = x + self.ffwd(x)
        return (x, enc) # need to include enc for sequential (next block)
    
class Encoder(nn.Module):
    """ Transformer encoder: a stack of Transformer blocks """

    def __init__(self, n_embd, n_head, n_layers):
        super().__init__()
        self.blocks = nn.ModuleList([EncoderBlock(n_embd, n_head) for _ in range(n_layers)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class Decoder(nn.Module):
    """ Transformer decoder: a stack of Transformer blocks """

    def __init__(self, n_embd, n_head, n_layers):
        super().__init__()
        self.blocks = nn.ModuleList([DecoderBlock(n_embd, n_head) for _ in range(n_layers)])

    def forward(self, x, enc):
        for block in self.blocks:
            x, enc = block(x, enc)
        return x, enc
    
class Transformer(nn.Module):
    """ Transformer model: encoder + decoder """

    def __init__(self, n_embd, max_len, n_head, n_layer):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table_inpt = nn.Embedding(block_size, n_embd)
        self.position_embedding_table_prompt = nn.Embedding(max_len, n_embd)
        self.enc = Encoder(n_embd, n_head, n_layer)
        self.dec = Decoder(n_embd, n_head, n_layer)
        # self.enc = nn.Sequential(*[EncoderBlock(n_embd, n_head=n_head) for _ in range(n_layer)])
        # self.dec = nn.Sequential(*[DecoderBlock(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, x, y):
        prompt, inpt = x
        prompt = self.token_embedding_table(prompt) + self.position_embedding_table_prompt(torch.arange(prompt.shape[1], device=prompt.device))
        inpt = self.token_embedding_table(inpt) + self.position_embedding_table_inpt(torch.arange(inpt.shape[1], device=inpt.device))
        enc_out = self.enc(prompt)
        dec_out, _ = self.dec(inpt, enc_out)
        logits = self.lm_head(self.ln_f(dec_out))
        
        if y is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
        
        return logits, loss
    
model = Transformer(n_embd, max_len, n_head, n_layer).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
  xb, yb = get_batch('train')
  logits, loss = model(xb, yb)
  
  if iter % 100 == 0: # fix this later so it uses better loss estimation
    print(loss.item())
  
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()