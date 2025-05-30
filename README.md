# Adversarial Alignment

Adversarial Alignment is an experimental research project focused on estimating the uncertainty of large language models (LLMs) from their internal activations during inference. The goal is to enable dynamic reasoning depth, scalable oversight, and introspective capabilities in LLMs.

The project is designed and developed independently by Leonhard Waibl as part of an AI alignment research initiative.

---

## ✨ Features

- Predicts model uncertainty using internal attention activations
- GNN-based encoding of attention maps
- Transformer encoder aggregation for token-level reasoning states
- Log-cosh loss for stable uncertainty regression
- Initial prototype focused on mathematical reasoning tasks

---

## 📊 Current Status

- LLM fine-tuning on math data complete (GPT-2 small)
- Attention extraction pipeline functional
- GNN + Transformer uncertainty model implemented
- Training shows decreasing loss; testing in progress
- Compute limitations: scaling experiments pending

---

## 🛠️ Technologies Used

- Python, PyTorch, Hugging Face Transformers, JAX, NumPy
- Graph Neural Networks (GNNs)
- Log-cosh loss for uncertainty regression

---

## 🔍 Research Focus

This project explores:
- Introspective model behavior
- Dynamic reasoning depth for LLMs
- Cost reduction through selective reasoning
- Scalable alignment tools for safe AI systems

---

## 📩 Contact

For feedback, ideas, or collaboration:
- Email: leonhardwaibl@gmail.com

---

This project is self-funded and developed independently as part of a broader research journey in AI alignment. Feedback and suggestions are welcome!

