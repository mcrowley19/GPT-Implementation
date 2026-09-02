import torch
import numpy as np
import gpt3_tokenizer
from datasets import load_dataset
import einops
from huggingface_hub import hf_hub_download

class Config:
    def __init__(self):
        self.block_size = 512
        self.batch_size = 16
        self.device = 'mps'
        self.nheads = 8
        self.dheads = 64
        self.d_model = self.nheads * self.dheads
        self.vocab = 8562
        self.W_u = torch.nn.Parameter(torch.rand(self.block_size, self.d_model)) * 0.02
        self.b_u = torch.nn.Parameter(torch.zeros(self.block_size))
    
class Head:
    def __init__(self, cfg):
        self.cfg = cfg
        W_pos = torch.nn.Parameter(torch.rand(cfg.block_size, cfg.d_model)) * 0.02
        W_Q = torch.nn.Parameter(torch.rand(cfg.d_model, cfg.d_model)) * 0.02
        W_K = torch.nn.Parameter(torch.rand(cfg.d_model, cfg.d_model)) * 0.02
        W_V = torch.nn.Parameter(torch.rand(cfg.d_model, cfg.d_model)) * 0.02
        b_q = torch.nn.Parameter(torch.zeros(cfg.d_model))
        b_k = torch.nn.Parameter(torch.zeros(cfg.d_model))
        b_v = torch.nn.Parameter(torch.zeros(cfg.d_model))
    
    def Attn(self, tokens):
        # I wil be getting a token matrix of dims (batch, tokens) and I want to result in (batch, tokens, modeldim)
        keys = einops.einsum(self.W_K, tokens, 'modeldim modeldim, batch tokens -> batch tokens modeldim') + self.b_k
        queries = einops.einsum(self.W_Q, tokens, 'modeldim modeldim, batch tokens -> batch tokens modeldim') + self.b_q

        kq = (keys @ queries) / self.cfg.dhead**0.5
        values = einops.einsum(self.cfg.W_V, tokens,'modeldim modeldim, batch tokens -> batch tokens modeldim') + self.cfg.b_v
        out = kq @ values
        return out
    
    def mask(mat):
        mask = torch.triu(mat[1:]) * float('-inf')
        res = torch.where(mask == 0, mat)
        return torch.nn.Softmax(res)
    
class LayerNorm(torch.nn.Module):
    def __init__(self, res, block_out):
        output = res + block_out
        return torch.nn.LayerNorm(output)

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, cfg):
        '''
        self.stack = torch.stack(
                        Head(cfg) * cfg.nheads
                    )
        '''
    def forward(self,resid):
        for head in self.stack:
            resid = resid + head(resid)
        
        return resid
    
class MLP(torch.nn.Module):
    def __init__(self, dim):
        self.W = torch.nn.Parameter(torch.random(dim))
        self.b = torch.nn.Parameter(torch.random(dim))
    def embed(self,resid):
        out = resid + ((resid @ self.W) + self.b)
        return out

class Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = Config()
        self.W_e = torch.nn.Parameter(torch.rand(self.cfg.vocab, self.cfg.d_model)) * 0.02
        self.W_pos = torch.nn.Parameter(torch.rand(self.cfg.vocab, self.cfg.d_model)) * 0.02
        self.b_e = torch.nn.Parameter(torch.zeros(self.cfg.block_size))
        self.multi = MultiHeadAttention(self.cfg)
    
    def embed(self, tokens):
        # We are getting tokens in the form batchnum, tokenlen and after we embed it should be batchnum,tokenlen,modeldim
        emb = self.W_e[tokens] + self.b_e
        # emb is size (batchnum,tokenlen,modeldim)
        pos = torch.arange(tokens.shape[1])
        pos_emb = self.W_pos[pos]
        print(pos_emb.shape)
        return emb + pos_emb

    def unembed(self, tokens):
        return self.M_u[tokens] + self.b_u

    def forward(self):
        x,y = self.get_batch()
        x = torch.tensor(gpt3_tokenizer.encode(x))
        y = torch.tensor(gpt3_tokenizer.encode(y))


        self.multi(x)


''''
My code for tokenizing. Got rid of it because it is faster to import pre-tokenized data

path = hf_hub_download(
    repo_id="hemantvirmani/gpt-training-dataset",
    filename="dataset.txt",
    repo_type="dataset"
)

with open(path, 'r') as f:
    data = f.read()


tokens = gpt3_tokenizer.encode(data)
'''

block_size = 512
batch_size = 16
tokens = np.memmap('train.bin', dtype=np.uint16, mode="r")
tokens = torch.from_numpy(tokens.astype(np.int64))

ix = torch.randint(0, len(tokens) - block_size, (batch_size,))
x = torch.stack([tokens[i:i+block_size] for i in ix])
y = torch.stack([tokens[i+1:i+block_size+1] for i in ix])


model = Transformer()

# x is (16x512) where we have context length of 16 tokens and 512 batches
print(x.shape)

print(model.embed(x).shape)


   






# First, I am going to need to find a way to convert the text dataset into tokens - Done
# Then, we need to run it through an embedding matrix (We index the embedding matrix with the tokens) 
# I will need to get key, query and values for each
# I can do this by multiplying them by a key matrix and a query matrix and adding biases
# I then multiply the key and query matrices and then multiply them by the value matrix at the end I think???
# I then add this onto the residual stream again at the end. 
# We then run this residual stream through an MLP and add it at the end again
# I then do this for multiple attention heads
# At the end we will run it through an unembedding matrix and we will use softmax to get a probability distribution
# Then we can use something like beam search to select what the next token is

# We will need to run this through a form of training loop that calculates the MSE loss.
# After this, maybe we can add a KV cache or something


