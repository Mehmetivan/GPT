# GPT
A mini version of GPT model implementing the Transformer model by the paper "Attention Is All You Need"

Absolutely — here’s a short **paragraph-style documentation** you can paste directly into your GitHub repository's README or project description.

---

📄 `gpt.py` — Training Script

This script trains a simple bigram language model using PyTorch. It reads a text file (e.g., `poems.txt`), builds a vocabulary of unique characters, and converts the input text into sequences of token indices. During training, the model learns to predict the next character in a sequence using randomly sampled batches. Every few hundred iterations, it evaluates average loss on both training and validation data. After training, the script uses the model to generate new text, starting from a single character and predicting one character at a time. It serves as a foundational example for understanding how autoregressive language models work.

---

🧠 `bigram.py` — Bigram Language Model

This file defines a minimal character-level language model where each character directly predicts the next one based on a learned embedding. The model uses a simple embedding layer (`nn.Embedding`) to map each input token to a vector of logits representing probabilities for the next token. It includes a `forward` method for loss computation and a `generate` method to sample new sequences autoregressively. This is a basic version of a language model designed to highlight core principles without the complexity of attention mechanisms or deep transformer layers.


