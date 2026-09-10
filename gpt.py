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
        self.train_steps = 7000
    
class Head(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_Q = torch.nn.Parameter(torch.randn(cfg.d_model, cfg.d_heads) * 0.02)
        self.W_K = torch.nn.Parameter(torch.randn(cfg.d_model, cfg.d_heads) * 0.02)
        self.W_V = torch.nn.Parameter(torch.randn(cfg.d_model, cfg.d_heads) * 0.02)
        self.b_q = torch.nn.Parameter(torch.zeros(cfg.d_heads))
        self.b_v = torch.nn.Parameter(torch.zeros(cfg.d_heads))
        self.b_k = torch.nn.Parameter(torch.zeros(cfg.d_heads))
        self.softmax = torch.nn.Softmax(dim=-1)
        self.k_cache = None
        self.v_cache = None
    
    def forward(self, tokens, cache):
        '''
        Here we have already embedded our tokens so they are of shape (batch, block_size, d_model). 
        The key, query and value matrices are of shape (d_model, d_head) and they project the tokens to the d_heads number of dimensions.
        This means that the output of the attention head is of shape (batch, block_size, dhead)
        '''
        
        keys = einops.einsum(self.W_K, tokens, 'modeldim dhead, batch new_tokens embedding -> batch new_tokens dhead') + self.b_k
        queries = einops.einsum(self.W_Q, tokens, 'modeldim dhead, batch tokens embedding -> batch tokens dhead') + self.b_q
        values = einops.einsum(self.W_V, tokens,'modeldim dhead, batch new_tokens embedding -> batch new_tokens dhead') + self.b_v

        if cache:
            if self.k_cache is not None:
                keys  = torch.cat((self.k_cache, keys), 1)
                values = torch.cat((self.v_cache,values),1 )
            self.k_cache = keys
            self.v_cache = values

        kq = (queries @ keys.swapdims(-1,-2)) / self.cfg.d_heads**0.5

        if tokens.shape[1] > 1:
            kq = self.mask(kq)
        

        kq_softmaxed = self.softmax(kq)
        out = kq_softmaxed @ values
        return out
    
    def mask(self, scores):
        mask = torch.triu(torch.ones(scores.shape[-1], scores.shape[-2], device=self.cfg.device), diagonal = 1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        res = scores + mask
        return res
    

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.stack = torch.nn.ModuleList([Head(cfg) for head in range(cfg.nheads)]) 
        self.cfg = cfg
        self.layer_norm = torch.nn.LayerNorm(cfg.d_model,device=cfg.device )
        self.W_O = torch.nn.Parameter(torch.randn(cfg.nheads * cfg.d_heads, cfg.d_model) * 0.02)
    def forward(self,resid, cache):
        outputs = []
        for head in self.stack:
            outputs.append(head(self.layer_norm(resid), cache))

       
        out = torch.cat(outputs, -1)
        out = out @ self.W_O
        return out
    
class MLP(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.W_e = torch.nn.Parameter(torch.randn(cfg.d_model, 4 * cfg.d_model) * 0.02)
        self.W_u = torch.nn.Parameter(torch.randn(4 * cfg.d_model, cfg.d_model) * 0.02)
        self.cfg = cfg
        self.layer_norm = torch.nn.LayerNorm(self.cfg.d_model, device=cfg.device)
        self.gelu = torch.nn.GELU()

    def forward(self,resid):
        resid_norm = self.layer_norm(resid)
        emb = einops.einsum(self.W_e, resid_norm, 'd_model d_mlp, batch tok_len d_model-> batch tok_len d_mlp')
        # now we convert x to have dimensions batch embedding d_mlp
        activation = self.gelu(emb)
        unemb =  einops.einsum(self.W_u, activation,  'd_mlp d_model, batch tok_len d_mlp -> batch tok_len d_model')
         
        return unemb

class TransformerBlock(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mha = MultiHeadAttention(cfg)
        self.mlp = MLP(cfg)

    def forward(self,resid, cache):
        resid = resid + self.mha(resid, cache)
        resid = resid + self.mlp(resid)
        return resid

class Model(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # I assume I can have W_e and W_u with the same dims because I heard that some papers used to use the same matrix for both
        self.W_e = torch.nn.Parameter(torch.randn(self.cfg.vocab, self.cfg.d_model) * 0.02)
        self.W_pos = torch.nn.Parameter(torch.randn(self.cfg.block_size, self.cfg.d_model) * 0.02)
        self.b_e = torch.nn.Parameter(torch.zeros(self.cfg.d_model))
        self.stack = torch.nn.ModuleList([TransformerBlock(cfg) for block in range(cfg.nlayers)]) 
        self.W_u = torch.nn.Parameter(torch.randn(self.cfg.vocab, self.cfg.d_model) * 0.02)
        self.b_u = torch.nn.Parameter(torch.zeros(self.cfg.vocab))
        self.layer_norm = torch.nn.LayerNorm(self.cfg.d_model, device=cfg.device)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.seen_token_count = 0
    def embed(self, tokens, cache):
        # B: The weight matrix is shape vocab_len x d_model of random numbers (Which are then tweaked through back prop). 
        # By indexing this with our tokens, we are picking out the rows that correspond to the tokens we have. This forms a (batch x block_size x d_model matrix)

        emb = self.W_e[tokens] + self.b_e

        offset = 0
        if cache:
            offset = self.seen_token_count
        pos = torch.arange(offset, offset + tokens.shape[1], device=self.cfg.device)
        pos_emb = self.W_pos[pos]
        return emb + pos_emb

    def forward(self, x, cache=False):
        resid = self.embed(x, cache)
        if cache:
            self.seen_token_count += x.shape[1]
        for tb in self.stack:
            resid = tb(resid, cache)

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

    def train_time(self, tokens):
        optimiser = torch.optim.AdamW(self.parameters(), lr= 1e-4)
        loss_func = torch.nn.CrossEntropyLoss()
        losses = []
        steps = []
        for step in tqdm(range(self.cfg.train_steps)):
            ix = torch.randint(0, len(tokens) - self.cfg.block_size, (self.cfg.batch_size,))
            x = torch.stack([tokens[i:i+self.cfg.block_size] for i in ix]).to(self.cfg.device)
            y = torch.stack([tokens[i+1:i+self.cfg.block_size+1] for i in ix]).to(self.cfg.device)


            logits = self(x, cache=False)
            optimiser.zero_grad()
            loss = loss_func(
                logits.reshape(-1, self.cfg.vocab),
                y.reshape(-1).long()
            )

            if step % 10 == 0:
                losses.append(loss.item())
                steps.append(step)
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
        import matplotlib.pyplot as plt
        plot = plt.plot(losses,steps)
        plot.set(xlabel='steps', ylabel='loss')
        plot.show()


        


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

if __name__ == "__main__":
    
    cfg = Config()
    tokens = np.memmap('train.bin', dtype=np.uint16, mode="r")
    tokens = torch.from_numpy(tokens.astype(np.int64))

    # A: We start by selecting batch_size batches of block_size tokens. We then stack these into a matrix (Forming a batch_size x block_size matrix)



    model = Model(cfg).to(cfg.device)
    model.train_time(tokens)
    torch.save(model.state_dict(), "michaelAI-5k-checkout.pt")

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

