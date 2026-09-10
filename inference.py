from gpt import Config, Model
import torch
cfg = Config()
model = Model(cfg).to(cfg.device)
model.load_state_dict(torch.load("michaelAI-5k-checkout.pt"))
model.eval()


import json
with open('vocab.json','r') as f:
    data = f.read()
    data = json.loads(data)


context = 512
input_str = "Criterion Collection Laserdisc in North America , and re- released in 1992 as "
with torch.no_grad():
    tokens = [torch.tensor([data.index(char)]) for char in input_str]
    token_tensor = torch.cat(tokens,dim=0).to(cfg.device)

    softmax = torch.nn.Softmax(dim=-1)

    logits = model(token_tensor.unsqueeze(0).to(cfg.device), cache = True)
    probs = torch.softmax(logits[0, -1], dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    output = token_tensor
    output = torch.cat([output, next_token], dim=-1)

    for iteration in range(context - len(input_str)):
        logits = model(next_token.unsqueeze(0).to(cfg.device), cache = True)
        probs = torch.softmax(logits[0, -1], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        output = torch.cat([output, next_token], dim=-1)


out_str = ''

for token in output:
    out_str += data[token]

print(out_str)
