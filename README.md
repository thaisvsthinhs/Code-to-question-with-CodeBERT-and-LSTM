# Code-to-Question Generation with CodeBERT + LSTM + Attention

This repository contains an implementation of a **code-to-question generation model** that uses a **pretrained CodeBERT encoder** and an **LSTM decoder with Bahdanau attention**. The model is trained to generate natural language questions from Python code snippets.

##  Model Architecture

- **Encoder**: [CodeBERT (microsoft/codebert-base)](https://huggingface.co/microsoft/codebert-base) encodes the input code into contextual embeddings of shape `(batch_size, seq_len, 768)`.
- **Decoder**: A multi-layer LSTM with attention takes the encoder outputs and generates the question token by token. It uses teacher forcing during training and greedy decoding during inference.

## Project Structure

1. **Data Preparation**: Code and question pairs are loaded and tokenized using CodeBERT tokenizer and a custom word-to-index mapping for target questions.
2. **Model Definition**: Combines pretrained CodeBERT as encoder with LSTM + Attention as decoder.
3. **Training**: The model is trained with cross-entropy loss and evaluated using BLEU and ROUGE.
4. **Evaluation**: Corpus-level BLEU and ROUGE metrics are used to evaluate the quality of the generated questions.

## Metrics

- **BLEU**: Measures precision-based n-gram overlap with reference questions.
- **ROUGE**: Measures recall and sequence-level overlap (LCS) between generated and reference questions.

## Requirements

- Python 3.8+
- PyTorch
- HuggingFace Transformers
- NLTK, rouge-score, tqdm, numpy
