# GPT
A mini version of GPT model implementing the Transformer model by the paper "Attention Is All You Need"

---

📄 `gpt.py` — Training Script

This script trains a simple bigram language model using PyTorch. It reads a text file (e.g., `poems.txt`), builds a vocabulary of unique characters, and converts the input text into sequences of token indices. During training, the model learns to predict the next character in a sequence using randomly sampled batches. Every few hundred iterations, it evaluates average loss on both training and validation data. After training, the script uses the model to generate new text, starting from a single character and predicting one character at a time. It serves as a foundational example for understanding how autoregressive language models work.

---

🧠 `bigram.py` — Bigram Language Model

This file defines a minimal character-level language model where each character directly predicts the next one based on a learned embedding. The model uses a simple embedding layer (`nn.Embedding`) to map each input token to a vector of logits representing probabilities for the next token. It includes a `forward` method for loss computation and a `generate` method to sample new sequences autoregressively. This is a basic version of a language model designed to highlight core principles without the complexity of attention mechanisms or deep transformer layers.


---

### 🧠 How It Works

#### 1. **Dataset Preparation**

The code starts by reading a text file (`poems.txt`) and builds a **character-level vocabulary**. Each character is assigned a unique integer ID for encoding (`stoi`) and decoding (`itos`).

* `encode(text)`: Converts a string to a list of integer IDs.
* `decode(list_of_ids)`: Converts a list of integer IDs back into a string.

The text is then split into:

* `train_data`: First 90% of the text.
* `val_data`: Remaining 10% used for validation.

#### 2. **Batching Logic**

The `get_batch(split)` function randomly extracts sequences from the dataset:

* `x`: A sequence of `block_size` characters (input).
* `y`: The next character at each position in `x` (target/output).

Each iteration samples a new batch, ensuring diverse learning signals during training.

#### 3. **Model Architecture**

The `BigramLanguageModel` is a very simple model that uses a **token embedding table** to predict the next character:

```python
self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
```

Each character index maps directly to a vector of logits, representing the unnormalized probabilities of the next character.

* **Forward pass**: Returns logits and computes cross-entropy loss.
* **Generate**: Given an initial context, it generates `n` new characters by sampling from the softmax distribution of the logits.

#### 4. **Training Process**

The model is trained using the **AdamW optimizer** and **cross-entropy loss**.

Every `eval_interval` (e.g., 300 steps), the training and validation loss is printed using the `estimate_loss()` function, which averages loss over several batches from both splits.

```python
logits, loss = model(xb, yb)
loss.backward()
optimizer.step()
```

#### 5. **Text Generation**

After training, the model generates new text using:

```python
context = torch.zeros((1, 1), dtype=torch.long, device=device)
decode(model.generate(context, max_new_tokens=500)[0].tolist())
```

It starts with a context (usually a blank or starting token) and samples characters autoregressively.

---

### ⚙️ How to Run This Project

#### ✅ Prerequisites

Install PyTorch first:

```bash
pip install torch
```

Ensure your Python environment has access to GPU if available (`cuda` support).

#### 📂 Setup

1. Clone this repository.
2. Place your training text file (`poems.txt`) in the same directory.

#### ▶️ Run the Training

Simply execute:

```bash
python gpt.py
```

This will:

* Train the model from scratch for `max_iters` (default: 3000)
* Print training and validation loss every 300 iterations
* Generate a sample of text from the trained model

#### ✏️ Modify Settings

You can adjust:

* `block_size`: Context window size
* `max_iters`: Number of training iterations
* `learning_rate`: Optimizer learning rate
* `batch_size`: Training batch size
* `eval_interval`: Print loss every N steps

---


