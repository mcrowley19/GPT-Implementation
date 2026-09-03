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
        self.nlayers = 6
    
class Head(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_pos = torch.nn.Parameter(torch.rand(cfg.block_size, cfg.d_model)) * 0.02
        self.W_Q = torch.nn.Parameter(torch.rand(cfg.vocab, cfg.d_model)) * 0.02
        self.W_K = torch.nn.Parameter(torch.rand(cfg.vocab, cfg.d_model)) * 0.02
        self.W_V = torch.nn.Parameter(torch.rand(cfg.vocab, cfg.d_model)) * 0.02
        self.b_q = torch.nn.Parameter(torch.zeros(cfg.d_model))
        self.b_v = torch.nn.Parameter(torch.zeros(cfg.d_model))
        self.b_k = torch.nn.Parameter(torch.zeros(cfg.d_model))
    
    def forward(self, tokens):
        '''
        What I think is going on here is we get our tokens tensor (batches, tokens, embedding), then we get the weight matrix (vocab_len, d_model).
        As we have a row for each possible token in the weight matrix, we pick out one value corresponding to each of our token embeddings.
        We then need to form a tensor that has the same amount of batches, has our choice of each embedding and also has another dimension representing
        info corresponding to the model dim
        '''
        keys = einops.einsum(self.W_K, tokens, 'vocab modeldim, batch tokens embedding -> batch modeldim embedding ') + self.b_k
        queries = einops.einsum(self.W_Q, tokens, 'vocab modeldim, batch tokens embedding -> batch modeldim embedding ') + self.b_q
        kq = (keys @ queries) / self.cfg.dheads**0.5

    
        masked_kq = self.mask(kq)
        values = einops.einsum(self.W_V, tokens,'vocab modeldim, batch tokens embedding -> batch modeldim embedding') + self.b_v
        out = masked_kq @ values
        return out
    
    def mask(self, scores):
        mask = torch.triu(torch.ones(scores.shape[-1], scores.shape[-2]), diagonal = 1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        res = scores + mask
        torch.nn.Softmax(res)
        return res
    

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.stack = torch.nn.ModuleList([Head(cfg) for head in range(cfg.nheads)]) 
        self.cfg = cfg

    def forward(self,resid):
        layer_norm = torch.nn.LayerNorm(self.cfg.block_size)
        out = resid.clone()
        for head in self.stack:
            out += resid + head(layer_norm(resid))
        return out
    
class MLP(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.W_e = torch.nn.Parameter(torch.rand(cfg.batch_size, cfg.block_size, 4 * cfg.d_model))
        self.W_u = torch.nn.Parameter(torch.rand(cfg.batch_size, cfg.block_size, cfg.d_model))

    def forward(self,resid):
        layer_norm = torch.nn.LayerNorm(self.cfg.block_size)
        resid_norm = layer_norm(resid)
        emb = einops.einsum(self.W_e, resid_norm, 'batch embedding d_mlp, batch d_model embedding -> batch embedding d_mlp')
        gelu = torch.nn.GELU()
        activation = gelu(emb)
        unemb =  einops.einsum(self.W_u, activation,  'batch embedding d_model, batch embedding d_mlp -> batch, embedding, d_model')
        return unemb + resid


class TransformerBlock(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mha = MultiHeadAttention(cfg)
        self.mlp = MLP(cfg)

    def forward(self,resid):
        resid = self.mha(resid)
        resid = self.mlp(resid)
        return resid



class Model(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # I assume I can have W_e and W_u with the same dims because I heard that some papers used to use the same matrix for both
        self.W_e = torch.nn.Parameter(torch.rand(self.cfg.vocab, self.cfg.d_model)) * 0.02
        self.W_pos = torch.nn.Parameter(torch.rand(self.cfg.vocab, self.cfg.d_model)) * 0.02
        self.b_e = torch.nn.Parameter(torch.zeros(self.cfg.block_size))
        self.stack = torch.nn.ModuleList([TransformerBlock(cfg) for block in range(cfg.nlayers)]) 
        self.W_u = torch.nn.Parameter(torch.rand(self.cfg.vocab, self.cfg.d_model)) * 0.02
        self.b_u = torch.nn.Parameter(torch.zeros(self.cfg.block_size))

    def embed(self, tokens):
        emb = self.W_e[tokens] + self.b_e
        pos = torch.arange(tokens.shape[1])
        pos_emb = self.W_pos[pos]
        return emb + pos_emb

    def unembed(self, tokens):
        return self.M_u[tokens] + self.b_u

    def forward(self, x, y):
        resid = self.embed(x)
        for tb in self.stack:
            resid += (resid)
        
        layer_norm = torch.nn.LayerNorm(self.cfg.block_size)
        resid_norm = layer_norm(resid)
        logits = einops.einsum(self.W_u, resid_norm, 'vocab d_model, batch embedding d_model  -> batch vocab embedding') + self.b_u
        return logits

        


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

cfg = Config()
model = Model(cfg)

print(model.forward(x,y))




# x is (16x512) where we have context length of 16 tokens and 512 batches



   
# Next to add:
#   - Training Loop
#   - Beam Search
#   - KV cache
#   - Custom tokenizer

