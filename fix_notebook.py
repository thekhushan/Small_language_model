import json

with open("Untitled-1.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

last_cell = notebook["cells"][-1]
source = [
    "import torch\n",
    "import torch.nn as nn\n",
    "from torch.nn import functional as F\n",
    "torch.manual_seed(1337)\n",
    "\n",
    "class BigramLanguageModel(nn.Module):\n",
    "\n",
    "    def __init__(self, vocab_size):\n",
    "        super().__init__()\n",
    "        # each token directly reads off the logits for the next token from a lookup table\n",
    "        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)\n",
    "\n",
    "    def forward(self, idx, targets=None):\n",
    "\n",
    "        # idx and targets are both (B,T) tensor of integers\n",
    "        logits = self.token_embedding_table(idx) # (B,T,C)\n",
    "\n",
    "        if targets is None:\n",
    "            loss = None\n",
    "        else:\n",
    "            B, T, C = logits.shape\n",
    "            logits = logits.view(B*T, C)\n",
    "            targets = targets.view(B*T)\n",
    "            loss = F.cross_entropy(logits, targets)\n",
    "\n",
    "        return logits, loss\n",
    "\n",
    "    def generate(self, idx, max_new_tokens):\n",
    "        # idx is (B, T) array of indices in the current context\n",
    "        for _ in range(max_new_tokens):\n",
    "            # get the predictions\n",
    "            logits, loss = self(idx)\n",
    "            # focus only on the last time step\n",
    "            logits = logits[:, -1, :] # becomes (B, C)\n",
    "            # apply softmax to get probabilities\n",
    "            probs = F.softmax(logits, dim=-1) # (B, C)\n",
    "            # sample from the distribution\n",
    "            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)\n",
    "            # append sampled index to the running sequence\n",
    "            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)\n",
    "        return idx\n",
    "\n",
    "vocab_size = len(chars)\n",
    "m = BigramLanguageModel(vocab_size)\n",
    "logits, loss = m(xb, yb)\n",
    "print(\"Initial loss:\", loss.item())\n",
    "\n",
    "optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)\n",
    "\n",
    "batch_size = 32\n",
    "for steps in range(1000):\n",
    "    # sample a batch of data\n",
    "    xb, yb = get_batch('train')\n",
    "\n",
    "    # evaluate the loss\n",
    "    logits, loss = m(xb, yb)\n",
    "    optimizer.zero_grad(set_to_none=True)\n",
    "    loss.backward()\n",
    "    optimizer.step()\n",
    "\n",
    "print(\"Final loss:\", loss.item())\n",
    "\n",
    "print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[0].tolist()))\n"
]

last_cell["source"] = source
# Clear outputs
last_cell["outputs"] = []

with open("Untitled-1.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
