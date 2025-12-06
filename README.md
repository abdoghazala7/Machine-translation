# 🧠 English-to-Arabic Transformer: A Deep Dive & Comparative Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![NLP](https://img.shields.io/badge/Domain-NLP-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📜 Overview

This project is not just a translation tool; it is a **comprehensive educational journey** into the mathematical heart of Natural Language Processing. The primary goal was to move beyond using Deep Learning libraries as "black boxes" and instead **build, train, and optimize a Transformer architecture from scratch**.

The project implements an **English-to-Arabic Neural Machine Translation (NMT)** system using the **CoVoST2** dataset (~270k samples). It features a rigorous comparative study between:
1.  **Custom Implementation:** A Transformer built manually from the ground up.
2.  **Baseline:** The standard `nn.Transformer` implementation provided by PyTorch.

> **🚀 Key Finding:** The Custom "From-Scratch" model significantly outperformed the PyTorch native implementation in translation quality, stability, and training efficiency.

🔗 **View Model Outputs & Weights:** [Kaggle Output Directory](https://www.kaggle.com/code/abdoghazala/machine-translation-en-ar/output)

---

## 🎯 Motivation

In the era of LLMs, understanding the foundational **"Attention Is All You Need"** architecture is crucial. The purpose of this project was to:
* Gain a granular understanding of **Self-Attention** and **Multi-Head Attention** mechanisms.
* Master the complexities of sequence-to-sequence training pipelines.
* Tackle the challenges of **Arabic NLP** (rich morphology and complex grammar).
* Demonstrate that a well-tuned custom architecture can outperform generic, out-of-the-box solutions.

---

## 🛠️ Tech Stack & Methodology

### 1. Architecture
* **Framework:** PyTorch.
* **Architecture:** Encoder-Decoder Transformer (Standard base configuration: $d_{model}=512, N=6, h=8$).
* **Tokenizer:** Byte-Pair Encoding (BPE) trained specifically for this corpus.

### 2. Key Techniques Implemented
* **Custom Layers:** Implemented Embeddings, Positional Encodings, Feed-Forward Networks, and Multi-Head Attention blocks manually.
* **Optimization:**
    * **Xavier (Glorot) Initialization:** Crucial for the stability of the custom model.
    * **Label Smoothing (0.1):** To prevent the model from becoming over-confident.
    * **Dynamic Padding & Masking:** Efficiently handling variable-length sequences.
    * **Scheduler:** Custom learning rate scheduler with warmup (inverse square root decay).
* **Metrics:** **BLEU Score** (Standard) and **ChrF Score** (Character n-gram F-score, highly effective for Arabic).

---

## 🏆 The Showdown: Scratch vs. PyTorch

We trained both models on the exact same dataset splits, hyperparameters, and hardware (NVIDIA P100 GPU). The results were decisive.

### 📊 Quantitative Results (Test Set)

| Metric | **Custom (From Scratch)** 🥇 | **PyTorch Native** 🥈 | **Improvement** |
| :--- | :---: | :---: | :---: |
| **BLEU Score** | **11.32** | 2.02 | **+460%** |
| **ChrF Score** | **47.26** | 23.52 | **+101%** |
| **Test Loss** | **3.64** | 5.08 | **-28%** (Lower is better) |
| **Training Time** | **~7 Hours** | ~8.5 Hours | **17% Faster** |
| **Convergence** | Reached optimum at **Epoch 30** | Stalled/Overfit at **Epoch 13** | Better Generalization |

### 📉 Analysis: Why did the Custom Model Win?

1.  **Weight Initialization:** The custom implementation utilized explicit **Xavier Uniform initialization**. This proved superior for this specific architecture depth and dataset size compared to the default initialization in `nn.Transformer`, leading to faster and more stable convergence.
2.  **Generalization vs. Overfitting:** The PyTorch baseline showed signs of overfitting early (around Epoch 13), where Validation Loss spiked while Training Loss decreased. The Custom model maintained a steady decrease in both losses, demonstrating superior generalization capabilities.
3.  **Architectural Transparency:** Building from scratch allowed for finer control over the internal flow of tensors, particularly in handling the `causal_mask` and `padding_mask` integration.

---

## 📂 Project Structure

```bash
├── config.py                   # Hyperparameters and file paths configuration
├── dataset.py                  # Custom PyTorch Dataset & Dynamic Masking logic
├── model.py                    # The Core: Transformer architecture implementation
├── train.py                    # Training loop, Validation, and Checkpointing
├── get_optimal_seq_lengths.py  # Statistical analysis to determine max_len
└── README.md                   # Project Documentation
```
## 🤝 Acknowledgements

* **Dataset:** Special thanks to **ymoslem** for providing the [CoVoST2-EN-AR-Text](https://huggingface.co/datasets/ymoslem/CoVoST2-EN-AR-Text) dataset on Hugging Face.
* **Inspiration:** The seminal paper *"Attention Is All You Need"* (Vaswani et al., 2017), which introduced the Transformer architecture.
