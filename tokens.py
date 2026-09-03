import numpy as np
from datasets import load_dataset

ds = load_dataset("hemantvirmani/gpt-training-dataset", split="train")
text = "\n".join(ds["text"])



chars = sorted(set(text))

# Dimensions for weight matrix
print(len(chars))
stoi = {value: index for index, value in enumerate(chars)}
ids = np.array([stoi[char] for char in text])

ids.tofile("train.bin")

import json
json.dump(chars, open("vocab.json", "w"))

