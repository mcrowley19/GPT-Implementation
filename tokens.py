import numpy as np
from datasets import load_dataset

ds = load_dataset("hemantvirmani/gpt-training-dataset", split="train")
text = "\n".join(ds["text"])
from collections import Counter

counts = Counter(text)

print(text[100000:100100])

chars = sorted(set(text))

# Dimensions for weight matrix
print(len(chars))
stoi = {value: index for index, value in enumerate(chars)}
ids = np.array([stoi[char] for char in text])

ids.tofile("train.bin")

import json
json.dump(chars, open("vocab.json", "w"))

