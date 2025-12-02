# Transformer from Scratch

A PyTorch implementation of a GPT-style transformer model trained on the Tiny Shakespeare dataset. This project demonstrates building a character-level language model from scratch using self-attention mechanisms.

## 🚀 Features

- **Multi-Head Self-Attention**: 6 attention heads with 64 dimensions each
- **Transformer Blocks**: 6 layers with residual connections and layer normalization
- **Character-level Generation**: Generates Shakespeare-like text at the character level

## 📋 Requirements

```bash
torch>=1.7.0
```

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/saugatadk/Transformer-from-scratch.git
cd Transformer-from-scratch
```

2. Install dependencies:
```bash
pip install torch
```

3. Download the Tiny Shakespeare dataset:
```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

## 🏃‍♂️ Usage

### Local Training

Run the training script:
```bash
python bigram_model.py
```

## ⚙️ Model Architecture

```
BigramLanguageModel(
  - Token Embedding: 65 → 384
  - Position Embedding: 256 → 384
  - 6 Transformer Blocks:
    - Multi-Head Attention (6 heads, 64 dims each)
    - Feed Forward Network (384 → 1536 → 384)
    - Layer Normalization + Residual Connections
  - Final Layer Norm
  - Output Projection: 384 → 65
)

Total Parameters: ~10.8M
```

## 🎛️ Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `batch_size` | 64 | Sequences processed in parallel |
| `block_size` | 256 | Maximum context length |
| `n_embd` | 384 | Embedding dimensions |
| `n_layers` | 6 | Number of transformer blocks |
| `n_head` | 6 | Number of attention heads |
| `learning_rate` | 3e-4 | Adam optimizer learning rate |
| `dropout` | 0.2 | Dropout probability |
| `max_iters` | 5000 | Training iterations |


## 🔬 Architecture Details

### Self-Attention Mechanism
- Scaled dot-product attention with causal masking
- Multi-head attention allows parallel focus on different aspects
- Residual connections and layer normalization for stable training

### Feed Forward Network
- Two linear layers with ReLU activation
- 4x expansion ratio (384 → 1536 → 384)
- Dropout for regularization


## 📚 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper


## 📄 License

This project is open source and available under the MIT License.
