import torch
import numpy as np
from datasets import load_dataset
import einops
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import time


class Config:
    def __init__(self):
        self.block_size = 512
        self.batch_size = 16
        self.device = 'mps'
        self.nheads = 8
        self.d_heads = 64
        self.d_model = self.nheads * self.d_heads
        self.vocab = 8562
        self.nlayers = 6
        self.train_steps = 1000
    
class Head(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_pos = torch.nn.Parameter(torch.rand(cfg.block_size, cfg.d_model) * 0.02)
        self.W_Q = torch.nn.Parameter(torch.rand(cfg.d_model, cfg.d_heads) * 0.02)
        self.W_K = torch.nn.Parameter(torch.rand(cfg.d_model, cfg.d_heads) * 0.02)
        self.W_V = torch.nn.Parameter(torch.rand(cfg.vocab, cfg.d_heads) * 0.02)
        self.b_q = torch.nn.Parameter(torch.zeros(cfg.d_heads))
        self.b_v = torch.nn.Parameter(torch.zeros(cfg.d_heads))
        self.b_k = torch.nn.Parameter(torch.zeros(cfg.d_heads))
        self.softmax = torch.nn.Softmax(dim=-1)
    
    def forward(self, tokens):
        '''
        Here we have already embedded our tokens so they are of shape (batch, block_size, d_model). 
        The key, query and value matrices are of shape (d_model, d_head) and they project the tokens to the d_heads number of dimensions.
        This means that the output of the attention head is of shape (batch, block_size, dhead)
        '''
        
        keys = einops.einsum(self.W_K, tokens, 'modeldim dhead, batch tokens embedding -> batch tokens dhead') + self.b_k
        queries = einops.einsum(self.W_Q, tokens, 'modeldim dhead, batch tokens embedding -> batch tokens dhead') + self.b_q
       
        kq = (keys @ queries.swapdims(-1,-2)) / self.cfg.d_heads**0.5
        masked_kq = self.mask(kq)
        values = einops.einsum(self.W_V, tokens,'modeldim dhead, batch tokens embedding -> batch tokens dhead') + self.b_v

        out = masked_kq @ values
        return out
    
    def mask(self, scores):
        mask = torch.triu(torch.ones(scores.shape[-1], scores.shape[-2], device=cfg.device), diagonal = 1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        res = scores + mask
        res = self.softmax(res)
        return res
    

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.stack = torch.nn.ModuleList([Head(cfg) for head in range(cfg.nheads)]) 
        self.cfg = cfg
        self.layer_norm = torch.nn.LayerNorm(cfg.d_model,device=cfg.device )
        self.W_O = torch.nn.Parameter(torch.rand(cfg.block_size, cfg.d_model) * 0.02)
    def forward(self,resid):
        outputs = []
        for head in self.stack:
            outputs.append(head(self.layer_norm(resid)))

       
        out = torch.cat(outputs, -1)
        out = out @ self.W_O
        return out
    
class MLP(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.W_e = torch.nn.Parameter(torch.rand(cfg.block_size, 4 * cfg.d_model))
        self.W_u = torch.nn.Parameter(torch.rand(cfg.block_size, cfg.d_model))
        self.cfg = cfg
        self.layer_norm = torch.nn.LayerNorm(self.cfg.d_model, device=cfg.device)
        self.gelu = torch.nn.GELU()

    def forward(self,resid):
        resid_norm = self.layer_norm(resid)
        emb = einops.einsum(self.W_e, resid_norm, 'embedding d_mlp, batch d_model embedding -> batch embedding d_mlp')
        # now we convert x to have dimensions batch embedding d_mlp
        activation = self.gelu(emb)
        unemb =  einops.einsum(self.W_u, activation,  'embedding d_model, batch embedding d_mlp -> batch embedding d_model')
         
        return unemb

class TransformerBlock(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mha = MultiHeadAttention(cfg)
        self.mlp = MLP(cfg)

    def forward(self,resid):
        resid = resid + self.mha(resid)
        resid = resid + self.mlp(resid)
        return resid

class Model(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # I assume I can have W_e and W_u with the same dims because I heard that some papers used to use the same matrix for both
        self.W_e = torch.nn.Parameter(torch.rand(self.cfg.vocab, self.cfg.d_model) * 0.02)
        self.W_pos = torch.nn.Parameter(torch.rand(self.cfg.vocab, self.cfg.d_model) * 0.02)
        self.b_e = torch.nn.Parameter(torch.zeros(self.cfg.block_size))
        self.stack = torch.nn.ModuleList([TransformerBlock(cfg) for block in range(cfg.nlayers)]) 
        self.W_u = torch.nn.Parameter(torch.rand(self.cfg.vocab, self.cfg.d_model) * 0.02)
        self.b_u = torch.nn.Parameter(torch.zeros(self.cfg.vocab))
        self.layer_norm = torch.nn.LayerNorm(self.cfg.d_model, device=cfg.device)
        self.softmax = torch.nn.Softmax(dim=-1)
    def embed(self, tokens):
        # B: The weight matrix is shape vocab_len x d_model of random numbers (Which are then tweaked through back prop). 
        # By indexing this with our tokens, we are picking out the rows that correspond to the tokens we have. This forms a (batch x block_size x d_model matrix)

        emb = self.W_e[tokens] + self.b_e
       
        pos = torch.arange(tokens.shape[1], device=self.cfg.device)
        pos_emb = self.W_pos[pos]
        return emb + pos_emb

    def unembed(self, tokens):
        return self.M_u[tokens] + self.b_u

    def forward(self, x):
        resid = self.embed(x)
        for tb in self.stack:
            resid = tb(resid)

        resid_norm = self.layer_norm(resid)
        # C: Our resid maintains the same shape as it originally did after we embedded it in Model.embed()
        # We now have to matrix multiply so that we have a logit for each token, for each element in the block, for each batch

        # Now we convert x to be of size batch number, items in each batch, tokens in vocabularly so we can get our tokens
        logits = einops.einsum(self.W_u, resid_norm, 'vocab d_model, batch block_size d_model  -> batch block_size vocab') + self.b_u
        #D: Each logit corresponds to a token. Here I will greedily just take the highest value logit and return it for each
        #logits = self.softmax(logits)
        #logit_indices = torch.argmax(logits, dim=-1)
        #return logit_indices
        return logits

    def train_time(self, x, y):
        optimiser = torch.optim.AdamW(self.parameters(), lr= 1e-4)
        loss_func = torch.nn.CrossEntropyLoss()
        for step in tqdm(range(self.cfg.train_steps)):
            logits = self(x)
            optimiser.zero_grad()
            loss = loss_func(
                logits.reshape(-1, self.cfg.vocab),
                y.reshape(-1).long()
            )

            if step % 10 == 0:
                print(f"Step {step} Loss: {loss.item()}")
            loss.backward()
            optimiser.step()
            
            '''
            print("Post Forward: ",t1 - start)
            print("Post Loss: ",t2 - start)
            print("Post Backward: ",t3 - start)
            print("Post optimiser",t4 - start)
            '''

        print(loss.item())


        


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


cfg = Config()

tokens = np.memmap('train.bin', dtype=np.uint16, mode="r")
tokens = torch.from_numpy(tokens.astype(np.int64))

# A: We start by selecting batch_size batches of block_size tokens. We then stack these into a matrix (Forming a batch_size x block_size matrix)
ix = torch.randint(0, len(tokens) - cfg.block_size, (cfg.batch_size,))
x = torch.stack([tokens[i:i+cfg.block_size] for i in ix])
y = torch.stack([tokens[i+1:i+cfg.block_size+1] for i in ix])


model = Model(cfg).to(cfg.device)
model.train_time(x.to(cfg.device),y.to(cfg.device))

'''
import json
with open('vocab.json','r') as f:
    data = f.read()
    data = json.loads(data)

for token in out:
    print(data[token])
'''






   
# Next to add:
#   - Training Loop
#   - Beam Search
#   - KV cache
#   - Custom tokenizer

